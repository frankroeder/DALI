import sys
import logging
import os
import warnings
from functools import partial
import matplotlib.pyplot as plt

import dreamerv3
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from carl.envs.carl_env import CARLEnv
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj
import jax
from dreamerv3.nets import Linear
from dreamerv3 import jaxutils

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
    _TASK2ENV,
    create_wrapped_carl_env,
)

cast = jaxutils.cast_to_compute
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

logging.captureWarnings(True)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"


def generate_envs(config, ctx_id):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    context_name = _TASK2CONTEXTS[task][ctx_id]["context"]
    print("USING CONTEXT:", context_name)
    env_cls: CARLEnv = _TASK2ENV[task]
    context = env_cls.get_default_context()
    ctor = lambda: create_wrapped_carl_env(env_cls, contexts={0: context}, config=config)
    ctor = partial(embodied.Parallel, ctor, "process")
    envs = [ctor()]
    return embodied.BatchEnv(envs, parallel=True), {context_name: context}


def _wrap_dream_agent(agent, z_dim, num_steps=50):
    def gen_dream(data):
        data = agent.preprocess(data)
        wm = agent.wm
        has_embed = "embed" in agent.config.ctx_encoder.inputs
        seq_len = min(data["obs"].shape[1], num_steps)
        ctx = wm.ctx_encoder(
            {
                "obs": data["obs"][:, :seq_len],
                "action": data["action"][:, :seq_len],
                "embed": wm.encoder(data)[:, :seq_len] if has_embed else None,
            }
        )

        # Posterior warm start
        post, _ = wm.rssm.observe(
            wm.encoder(data)[:, :seq_len],
            data["action"][:, :seq_len],
            data["is_first"][:, :seq_len],
            dcontext=ctx[:, :seq_len],
        )

        # action = data["action"][:, :seq_len]
        # start = {k: v[:, 0] for k, v in post.items()}
        # # Generate the baseline imagined trajectory using the original actions and context.
        # baseline_imagined = wm.rssm.imagine(action, start, dcontext=ctx)
        # baseline_reconst = wm.heads["decoder"]({**baseline_imagined, "context": ctx})
        # baseline_img = baseline_reconst["obs"].mode()

        # report = {"baseline_actions": action}

        def rollout(context_tensor):
            # Initialize at t=0 without time axis
            prev_stoch = post["stoch"][:, :1]
            prev_deter = post["deter"][:, :1]
            policy_state = agent.policy_initial(1)[1]

            act_info, policy_state = agent.task_behavior.policy(
                {"stoch": prev_stoch, "deter": prev_deter},
                policy_state,
                dcontext=context_tensor[:, :1],
            )
            act = act_info["action"].mode()

            action_seq = []
            for t in range(num_steps):
                dctx = context_tensor[:, t: t + 1]
                if wm.rssm._action_clip > 0.0:
                    act = act * sg(
                        wm.rssm._action_clip / jnp.maximum(wm.rssm._action_clip, jnp.abs(act))
                    )

                if wm.rssm._classes:
                    shape = prev_stoch.shape[:-2] + (wm.rssm._stoch * wm.rssm._classes,)
                    prev_stoch = prev_stoch.reshape(shape)
                if len(act.shape) > len(prev_stoch.shape):  # 2D actions.
                    shape = act.shape[:-2] + (np.prod(act.shape[-2:]),)
                    act = act.reshape(shape)

                # RSSM prior step
                x = jnp.concatenate([prev_stoch, act], -1)
                x = x if dctx is None else jnp.concatenate([x, dctx], -1)
                if wm.rssm._add_context_prior:
                    x = jnp.concatenate([x, dctx], -1)
                x = wm.rssm.get("img_in", Linear, **wm.rssm._kw)(x)
                x, new_deter = wm.rssm._gru(x, prev_deter)
                if wm.rssm._add_context_prior:
                    x = jnp.concatenate([x, dctx], -1)
                x = wm.rssm.get("img_out", Linear, **wm.rssm._kw)(x)
                stats = wm.rssm._stats("img_stats", x)
                dist = wm.rssm.get_dist(stats)
                # new_stoch = dist.mode()
                new_stoch = dist.sample(seed=nj.rng())
                action_seq.append(act)

                prev_stoch, prev_deter = new_stoch, new_deter
                act_info, policy_state = agent.task_behavior.policy(
                    {"stoch": prev_stoch, "deter": prev_deter}, policy_state, dcontext=dctx
                )
                act = act_info["action"].mode()
            return jnp.stack(action_seq, axis=1)

        baseline_actions = rollout(ctx)
        sigma = ctx[0].std(0)[z_dim]
        mags = jnp.linspace(-3, 3, 7)
        report = {"baseline_actions": baseline_actions, "magnitudes": mags}
        for i, m in enumerate(mags):
            ctx_p = ctx.at[:, :, z_dim].add(m * sigma)
            cf_actions = rollout(ctx_p)
            report[f"delta_z{z_dim}_m{i}"] = cf_actions - baseline_actions
        return report

    return gen_dream


def record_dream(agent, env, args, ctx_info, logdir, dream_fn, ctx_id, z_dim):
    def per_ep(ep):
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jbatch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report = dream_fn(agent.varibs, rng, jbatch)
        report = agent._convert_mets(report, agent.train_devices)

        out = logdir / f"action_traversal_ctx{ctx_id}_z{z_dim}"
        out.mkdirs()

        baseline = report[0]["baseline_actions"][0,:, 0]
        mags = report[0]["magnitudes"]
        dims = baseline.shape[-1]
        times = baseline.shape[0]

        fig, axs = plt.subplots(dims, 1, figsize=(8, 4 * dims))
        for d in range(dims):
            ax = axs[d]
            ax.plot(range(times), baseline[:, d], label="Baseline", linewidth=2)
            for i, m in enumerate(mags):
                delta = report[0][f"delta_z{z_dim}_m{i}"][0, :, 0, d]
                ax.plot(range(times), delta, label=f"m={m:.1f}", alpha=0.6)
            ax.set_title(f"Action Dim {d}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(out / f"actions_z{z_dim}_ctx{ctx_id}.pdf")
        plt.close()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, _: per_ep(ep))
    ck = embodied.Checkpoint()
    ck.agent = agent
    ck.load(args.from_checkpoint, keys=["agent"])
    driver(lambda *a: agent.policy(*a, mode="eval"), episodes=1)


def main():
    warnings.filterwarnings("ignore")
    parsed, other = embodied.Flags(logdir="", episodes=1, ctx_id=0, z_dim=0).parse_known()
    logdir = embodied.Path(parsed.logdir)
    ctx_id, z_dim = parsed.ctx_id, parsed.z_dim

    cfg = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    cfg["envs"]["amount"] = 1
    config = embodied.Config(cfg)
    config = embodied.Flags(config).parse(other)

    ckpt = logdir / "checkpoint.ckpt"
    assert ckpt.exists()
    config = config.update({"run.from_checkpoint": str(ckpt)})

    c = embodied.Checkpoint()
    c.step = embodied.Counter()
    c.load(str(ckpt), keys=["step"])
    step = c._values["step"]

    env, ctx_info = generate_envs(config, ctx_id)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    dream_fn = nj.jit(nj.pure(_wrap_dream_agent(agent.agent, z_dim)), device=agent.train_devices[0])
    args = embodied.Config(
        **config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length
    )
    record_dream(agent, env, args, ctx_info, logdir, dream_fn, ctx_id, z_dim)
    env.close()


if __name__ == "__main__":
    main()
