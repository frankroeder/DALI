import sys
import logging
import os
import warnings
from functools import partial
import numpy as np
import ruamel.yaml as yaml
import jax
import jax.numpy as jnp
import dreamerv3
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj
import pickle

from carl.envs.carl_env import CARLEnv
from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    create_wrapped_carl_env,
)

logging.captureWarnings(True)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

tree_map = jax.tree_util.tree_map

def generate_envs(config):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    context = env_cls.get_default_context()
    envs = []
    ctor = lambda: create_wrapped_carl_env(
        env_cls, contexts={0: context}, config=config
    )
    ctor = partial(embodied.Parallel, ctor, "process")
    envs = [ctor() for _ in range(20)]
    return embodied.BatchEnv(envs, parallel=True), context

def _wrap_dream_agent(agent):
    def gen_dream(data):
        data = agent.preprocess(data)
        data = tree_map(lambda x: x[:, :-1].reshape([10, -1] + list(x.shape[2:])), data)
        wm = agent.wm
        seq_len = agent.config.batch_length
        has_embed = "embed" in agent.config.ctx_encoder.inputs

        _ctx = wm.ctx_encoder({
                "obs": data["obs"][:, :seq_len],
                "action": data["action"][:, :seq_len],
                "embed": wm.encoder(data)[:, :seq_len] if has_embed else None
        })
        if isinstance(_ctx, dict):
            _ctx_samp = _ctx["dist"].sample(seed=nj.rng())
            ctx = jnp.stack([_ctx_samp for _ in range(seq_len)], axis=1)
        else:
            ctx = _ctx

        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :seq_len],
            data["action"][:, :seq_len],
            data["is_first"][:, :seq_len],
            dcontext=ctx[:, :seq_len],
        )
        start = {k: v[:, 0] for k, v in posterior_states.items()}
        # action = jnp.zeros_like(data["action"][:, :seq_len])
        action = data["action"][:, :seq_len]
        baseline_imagined = wm.rssm.imagine(action, start, dcontext=ctx)
        baseline_reconst = wm.heads["decoder"]({**baseline_imagined, "context": ctx})
        if "image" in baseline_reconst.keys():
            baseline_img = baseline_reconst["image"].mode()
        else:
            baseline_img = baseline_reconst["obs"].mode()
        report = {}
        for z_dim in range(8):
            sigma_z = ctx.reshape(-1, 8).std(0)[z_dim]
            # delta = jax.random.uniform(nj.rng(), shape=(seq_len,), dtype=ctx.dtype, minval=sigma_z/2, maxval=sigma_z * 3)
            # sign = jax.random.choice(nj.rng(), jnp.array([-1, 1]), shape=(seq_len,))
            # ctx_perturbed = ctx.at[:, :, z_dim].add(delta * sign)

            # ctx_perturbed = ctx.at[:, :, z_dim].add(2.0 * sigma_z)

            sign = jax.random.choice(nj.rng(), jnp.array([-1, 1]), shape=(seq_len,))
            ctx_perturbed = ctx.at[:, :, z_dim].add(sign * 3.0 * sigma_z)

            counterfactual_imagined = wm.rssm.imagine(
                action, start, dcontext=ctx_perturbed
            )
            counterfactual_imagined = {
                **counterfactual_imagined,
                "context": ctx_perturbed,
            }
            counterfactual_reconst = wm.heads["decoder"](counterfactual_imagined)
            if "image" in baseline_reconst.keys():
                cf_img = counterfactual_reconst["image"].mode()
            else:
                cf_img = counterfactual_reconst["obs"].mode()
            key = f"z_dim{z_dim}"
            report[key] = {
                "original": baseline_img,
                "imagined": cf_img
            }
        return report
    return gen_dream

def record_dream(agent, env, args, ctx_info, logdir, dream_agent_fn, task,episodes):
    big_dataset = {"original": {}, "imagined": {}}

    def per_episode(ep):
        nonlocal agent, big_dataset, ctx_info, env
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)
        for key in report.keys():
            big_dataset["original"].setdefault(key, []).append(report[key]["original"])
            big_dataset["imagined"].setdefault(key, []).append(report[key]["imagined"])

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)
    return {
            key: {
            _key: np.concatenate(_val)
                  for _key, _val in val.items()
        }
        for key, val in big_dataset.items()
    }

def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # Parse only logdir and episodes
    parsed, other = embodied.Flags(logdir="", episodes=150).parse_known()
    logdir = embodied.Path(parsed.logdir)

    # Load configuration
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config["envs"]["amount"] = 20
    config = embodied.Config(config)
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Load step counter
    ckpt = embodied.Checkpoint()
    ckpt.step = embodied.Counter()
    ckpt.load(checkpoint, keys=["step"])
    step = ckpt._values["step"]

    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite

    dream_agent_fn = None
    agent = None
    env, ctx_info = generate_envs(config)
    if agent is None:
        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
        dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )

    dataset = record_dream(
        agent,
        env,
        args,
        ctx_info,
        logdir,
        dream_agent_fn,
        task,
        parsed.episodes,
    )
    env.close()

    # dest_name = f"dataset_all_ep{parsed.episodes}.pkl"
    # dest_name = f"dataset_all_ep{parsed.episodes}_sigma_div_2_to_sigma_times_3.pkl"
    # dest_name = f"dataset_all_ep{parsed.episodes}_pos_2x_sigma.pkl"
    dest_name = f"dataset_all_ep{parsed.episodes}_pos_neg_3x_sigma.pkl"
    # Save the big dataset as a pkl file
    with open(logdir / dest_name, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {logdir / dest_name}")

if __name__ == "__main__":
    main()
