import sys
import logging
import os
import warnings
from functools import partial

import cv2
import dreamerv3
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from carl.envs.carl_env import CARLEnv
from dreamerv3 import embodied, jaxutils
from dreamerv3 import ninjax as nj
from dreamerv3.embodied.core.logger import _encode_gif

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
    env_cls: CARLEnv = _TASK2ENV[task]
    context = env_cls.get_default_context()
    # take the longest property
    # ctx_val = _TASK2CONTEXTS[task][ctx_id]["extrapolate_double"][-1]
    # shortest
    ctx_val = _TASK2CONTEXTS[task][ctx_id]["extrapolate_double"][-2]
    context[context_name] = ctx_val

    envs = []
    ctor = lambda: create_wrapped_carl_env(
        env_cls, contexts={0: context}, config=config
    )
    ctor = partial(embodied.Parallel, ctor, "process")
    envs = [ctor()]
    return embodied.BatchEnv(envs, parallel=True), context


def _wrap_dream_agent(agent):
    def gen_dream(data):
        data = agent.preprocess(data)
        wm = agent.wm
        embed = None

        report = {}
        for dim in range(agent.config.ctx_encoder.linear_ctx_out.hunits):
            for pertubation in np.linspace(-10.0, 10.0, 5):
                embed = (
                    wm.encoder(data)
                    if "embed" in agent.config.ctx_encoder.inputs
                    else None
                )
                if "embed" in agent.config.ctx_encoder.inputs:
                    data["embed"] = embed
                ctx = wm.ctx_encoder({**data})
                seq_len = 10
                # apply pertubation
                ctx = ctx.at[0, :, dim].set(ctx[0, :, dim] + pertubation)

                state = wm.initial(len(data["is_first"]))
                # report.update(wm.loss(data, state)[-1][-1])
                posterior_states, _ = wm.rssm.observe(
                    wm.encoder(data)[:, :seq_len],
                    data["action"][:, :seq_len],
                    data["is_first"][:, :seq_len],
                    dcontext=ctx[:, :seq_len],
                )
                start = {k: v[:, -1] for k, v in posterior_states.items()}
                posterior_states = {**posterior_states, "context": ctx[:, :seq_len]}

                # when we decode posterior frames, we are just reconstructing as we
                # have the ground truth observations used for infering the latents
                # posterior_reconst = wm.heads["decoder"](posterior_states)
                # posterior_cont = wm.heads["cont"](posterior_states)

                posterior_reconst_5 = wm.heads["decoder"](posterior_states)
                posterior_cont_5 = wm.heads["cont"](posterior_states)

                imagined_states = wm.rssm.imagine(
                    data["action"][:, :seq_len],
                    start,
                    dcontext=ctx[:, :seq_len],
                )
                imagined_states = {**imagined_states, "context": ctx[:, :seq_len]}
                imagine_reconst = wm.heads["decoder"](imagined_states)
                imagine_cont = wm.heads["cont"](imagined_states)

                report[f"terminate_dim{dim}_{pertubation:0.2f}"] = (
                    1
                    - jnp.concatenate(
                        [posterior_cont_5.mode(), imagine_cont.mode()], 1
                    )[0]
                )
                # report["terminate_post"] = 1 - posterior_cont.mode()[0]

                # if "context" in data and "context" in posterior_reconst_5:
                #     model_ctx = jnp.concatenate(
                #         [
                #             posterior_reconst_5["context"].mode(),
                #             imagine_reconst["context"].mode(),
                #         ],
                #         1,
                #     )[
                #         ..., 1:
                #     ]  # pick only the length dimension of the context
                #     truth = data["context"][..., 1:]
                #     report["ctx"] = jnp.concatenate([truth, model_ctx], 2)
                #     # report["ctx_post"] = posterior_reconst["context"].mode()[..., 1:]
                truth = data["image"][:, : (seq_len * 2)].astype(jnp.float32)
                post_5_rest_imagined_reconst = jnp.concatenate(
                    [
                        posterior_reconst_5["image"].mode()[:, :seq_len],
                        imagine_reconst["image"].mode(),
                    ],
                    1,
                )
                error_1 = (post_5_rest_imagined_reconst - truth + 1) / 2
                # post_full_reconst = posterior_reconst["image"].mode()
                # error_2 = (post_full_reconst - truth + 1) / 2
                # video = jnp.concatenate(
                #     [truth, post_5_rest_imagined_reconst, error_1, post_full_reconst, error_2],
                #     2,
                # )
                video = jnp.concatenate(
                    [truth, post_5_rest_imagined_reconst, error_1],
                    2,
                )
                report[f"image_dim{dim}_{pertubation:0.2f}"] = jaxutils.video_grid(
                    video
                )
        return report

    return gen_dream


def record_dream(agent, env, args, ctx_info, logdir, dream_agent_fn, ctx_id, task):
    report = None

    def per_episode(ep):
        nonlocal agent, report
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)
        for dim in range(8):
            for pertubation in np.linspace(-10.0, 10.0, 5):
                video = report[f"image_dim{dim}_{pertubation:0.2f}"]

                video = np.clip(255 * video, 0, 255).astype(np.uint8)

                context_name = _TASK2CONTEXTS[task][ctx_id]["context"]
                l = ctx_info[context_name]

                fname = f"{l:0.2f}"
                path = logdir / (f"dreams_dim{dim}_{pertubation:0.2f}_{context_name}")
                path.mkdirs()

                for i in range(len(video)):
                    posterior = video[i]
                    cv2.imwrite(
                        str(path / f"{fname}_posterior_{i}.png"), posterior[:, :, ::-1]
                    )

                for i in range(len(video)):
                    if report[f"terminate_dim{dim}_{pertubation:0.2f}"][i] > 0:
                        # draw a line at the right end of the image
                        video[
                            i,
                            64:128,
                            -10:,
                        ] = [255, 0, 0]

                encoded_img_str = _encode_gif(video, 30)
                with open(path / f"{fname}.gif", "wb") as f:
                    f.write(encoded_img_str)
                # find the first terminate index
                if (
                    np.where(report[f"terminate_dim{dim}_{pertubation:0.2f}"] > 0)[
                        0
                    ].size
                    > 0
                ):
                    terminate_idx = np.where(
                        report[f"terminate_dim{dim}_{pertubation:0.2f}"] > 0
                    )[0][0]
                else:
                    terminate_idx = len(video)
                video = video[: min(max(terminate_idx + 1, 100), len(video))]
                # stack the video frames horizontally
                video = np.hstack(video)
                # draw a rectange around first 64 *5 pixels horizontally and 192 pixels vertically
                cv2.rectangle(video, (0, 0), (64 * 5, 192), (0, 128, 0), 2)
                # save the
                cv2.imwrite(str(path / f"{fname}.png"), video[:, :, ::-1])

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

    parsed, other = embodied.Flags(logdir="", episodes=1, ctx_id=1).parse_known()
    logdir = embodied.Path(parsed.logdir)
    ctx_id = parsed.ctx_id
    logdir = embodied.Path(parsed.logdir)
    # load the config from the logdir
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config["envs"]["amount"] = 1
    config = embodied.Config(config)
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Just load the step counter from the checkpoint, as there is
    # a circular dependency to load the agent.
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
        dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
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
        task,
    )
    env.close()


if __name__ == "__main__":
    main()
