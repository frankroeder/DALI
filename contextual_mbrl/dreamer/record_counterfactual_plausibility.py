import sys
import logging
import os
import warnings
from functools import partial

import cv2
import dreamerv3
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from carl.envs.carl_env import CARLEnv
from dreamerv3 import embodied, jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.embodied.core.logger import _encode_gif
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    create_wrapped_carl_env,
)

logging.captureWarnings(True)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def generate_envs(config, ctx_id):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    context_name = _TASK2CONTEXTS[task][ctx_id]["context"]
    print("USING CONTEXT:", context_name)
    env_cls: CARLEnv = _TASK2ENV[task]
    context = env_cls.get_default_context()
    envs = []
    ctor = lambda: create_wrapped_carl_env(env_cls, contexts={0: context}, config=config)
    ctor = partial(embodied.Parallel, ctor, "process")
    envs = [ctor()]
    return embodied.BatchEnv(envs, parallel=True), context


def _wrap_dream_agent(agent, z_dim):
    def gen_dream(data):
        # Preprocess incoming data
        data = agent.preprocess(data)
        wm = agent.wm
        embed = wm.encoder(data)
        seq_len = agent.config.batch_length

        # Compute the latent context
        _ctx = wm.ctx_encoder(
            {
                "obs": data["obs"][:, :seq_len],
                "action": data["action"][:, :seq_len],
                "embed": embed[:, :seq_len],
            }
        )
        if isinstance(_ctx, dict):
            _ctx_samp = _ctx["dist"].sample(seed=nj.rng())
            ctx = jnp.stack([_ctx_samp for _ in range(seq_len)], axis=1)
        else:
            ctx = _ctx

        # Compute posterior states and take the initial state
        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :seq_len],
            data["action"][:, :seq_len],
            data["is_first"][:, :seq_len],
            dcontext=ctx[:, :seq_len],
        )
        start = {k: v[:, 0] for k, v in posterior_states.items()}
        if agent.config.task == "carl_dmc_ball_in_cup":
            action = jnp.zeros_like(data["action"][:, :seq_len])
        else:
            action = data["action"][:, :seq_len]
        # Generate baseline trajectory
        baseline_imagined = wm.rssm.imagine(action, start, dcontext=ctx)
        baseline_reconst = wm.heads["decoder"]({**baseline_imagined, "context": ctx})
        baseline_img = baseline_reconst["image"].mode()

        report = {}
        # Define positive delta magnitudes
        sigma_z = ctx[0].std(0)[z_dim]
        num_steps = 30
        deltas = jnp.linspace(0, sigma_z * 3, num_steps)

        for idx, delta in enumerate(deltas):
            # Positive perturbation
            ctx_pos = ctx.at[:, z_dim].add(delta)
            counterfactual_imagined_pos = wm.rssm.imagine(action, start, dcontext=ctx_pos)
            counterfactual_reconst_pos = wm.heads["decoder"](
                {**counterfactual_imagined_pos, "context": ctx_pos}
            )
            cf_img_pos = counterfactual_reconst_pos["image"].mode()
            video = jnp.concatenate(
                [
                    baseline_img,
                    cf_img_pos,
                    (baseline_img - cf_img_pos + 1) / 2,
                ],
                axis=2,
            )
            key = f"counterfactual_dim{z_dim}_step{idx}"
            report[key] = jaxutils.video_grid(video)
        return report

    return gen_dream


def record_dream(agent, env, args, ctx_info, logdir, dream_agent_fn, ctx_id, z_dim, task):
    report = None

    def per_episode(ep):
        nonlocal agent, report
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)

        for key, video in report.items():
            video = np.clip(255 * np.array(video), 0, 255).astype(np.uint8)
            path = logdir / "img_counterfactuals" / key
            path.mkdirs()

            # Save individual frames as PNGs
            for i in range(video.shape[0]):
                if i % 5 == 0:
                    frame = video[i]
                    cv2.imwrite(str(path / f"{key}_frame_{i}.png"), frame[:, :, ::-1])

            # Create GIF
            with imageio.get_writer(str(path / f"{key}.gif"), mode="I", fps=30) as writer:
                for frame in video:
                    writer.append_data(frame)

            # Save overview image
            if key != "baseline":
                overview = np.concatenate(video[::5], axis=1)
                cv2.imwrite(str(path / f"{key}_overview.png"), overview[:, :, ::-1])
            else:
                cv2.imwrite(str(path / f"{key}.png"), video[0][:, :, ::-1])

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=1)


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")
    parsed, other = embodied.Flags(logdir="", episodes=1, ctx_id=0, z_dim=3).parse_known()
    logdir = embodied.Path(parsed.logdir)
    ctx_id = parsed.ctx_id
    z_dim = parsed.z_dim
    logdir = embodied.Path(parsed.logdir)
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config["envs"]["amount"] = 1
    config = embodied.Config(config)
    config = embodied.Flags(config).parse(other)
    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})
    ckpt = embodied.Checkpoint()
    ckpt.step = embodied.Counter()
    ckpt.load(checkpoint, keys=["step"])
    step = ckpt._values["step"]
    suite, task = config.task.split("_", 1)
    dream_agent_fn = None
    agent = None
    env, ctx_info = generate_envs(config, ctx_id)
    if agent is None:
        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent, z_dim))
        dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    record_dream(
        agent,
        env,
        args,
        ctx_info,
        logdir,
        dream_agent_fn,
        ctx_id,
        z_dim,
        task,
    )
    env.close()


if __name__ == "__main__":
    main()
