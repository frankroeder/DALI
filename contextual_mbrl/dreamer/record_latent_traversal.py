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
        # Preprocess incoming data.
        data = agent.preprocess(data)
        wm = agent.wm

        # use only a short sequence
        seq_len = 50  # agent.config.batch_length
        has_embed = "embed" in agent.config.ctx_encoder.inputs

        # Compute the latent context from the data via the frozen context encoder.
        # (ctx represents the latent variables encoding contextual factors.)
        ctx = wm.ctx_encoder(
            {
                "obs": data["obs"][:, :seq_len],
                "action": data["action"][:, :seq_len],
                "embed": wm.encoder(data)[:, :seq_len] if has_embed else None,
            }
        )

        # Compute the posterior states from the first seq_len timesteps and take the last timestep.
        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :seq_len],
            data["action"][:, :seq_len],
            data["is_first"][:, :seq_len],
            dcontext=ctx[:, :seq_len],
        )
        start = {k: v[:, 0] for k, v in posterior_states.items()}

        action = jnp.zeros_like(data["action"][:, :seq_len])
        # action = data["action"][:, :seq_len]

        # Generate the baseline imagined trajectory using the original actions and context.
        baseline_imagined = wm.rssm.imagine(action, start, dcontext=ctx)
        baseline_reconst = wm.heads["decoder"]({**baseline_imagined, "context": ctx})
        baseline_img = baseline_reconst["obs"].mode()

        report = {"original": baseline_img}

        # For each latent dimension and perturbation value, generate a counterfactual trajectory.
        # noise_scale = 0.1  # noise level to apply to actions
        sigma_z = ctx[0].std(0)[z_dim]
        for midx, magnitude in enumerate(jnp.linspace(-3, 3, 7)):
            # Perturb the context: add delta only to the j-th dimension.
            ctx_perturbed = ctx.at[:, :, z_dim].add(magnitude * sigma_z)
            counterfactual_imagined = wm.rssm.imagine(action, start, dcontext=ctx_perturbed)

            prev_latent = {
                "stoch": posterior_states["stoch"][:, :1],
                "deter": posterior_states["deter"][:, :1],
                "logit": posterior_states["logit"][:, :1],
            }
            dcontext = ctx_perturbed[:, :1]
            imagined_data ={
                "context": [dcontext],
                "deter": [prev_latent["deter"]],
                "stoch": [prev_latent["stoch"]],
                "logit": [prev_latent["logit"]],
            }
            task_action, task_state = agent.task_behavior.policy(prev_latent, agent.policy_initial(1)[1], dcontext=dcontext)
            prev_action = task_action["action"].sample(seed=nj.rng())
            for step in range(49):
                prev_stoch = prev_latent["stoch"]
                # prev_action = cast(data["action"][:, :1])
                if wm.rssm._action_clip > 0.0:
                    prev_action *= sg(
                        wm.rssm._action_clip / jnp.maximum(wm.rssm._action_clip, jnp.abs(prev_action))
                    )
                if wm.rssm._classes:
                    shape = prev_stoch.shape[:-2] + (wm.rssm._stoch * wm.rssm._classes,)
                    prev_stoch = prev_stoch.reshape(shape)
                if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
                    shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
                    prev_action = prev_action.reshape(shape)
                x = jnp.concatenate([prev_stoch, prev_action], -1)
                x = x if dcontext is None else jnp.concatenate([x, dcontext], -1)
                # print(f"X IS {x.shape} {dcontext}")
                x = wm.rssm.get("img_in", Linear, **wm.rssm._kw)(x)
                x, deter = wm.rssm._gru(x, prev_latent["deter"])
                if dcontext is not None and wm.rssm._add_context_prior:
                    x = jnp.concatenate([x, dcontext], -1)
                x = wm.rssm.get("img_out", Linear, **wm.rssm._kw)(x)
                stats = wm.rssm._stats("img_stats", x)
                dist = wm.rssm.get_dist(stats)
                stoch = dist.sample(seed=nj.rng())
                prior = {"stoch": stoch, "deter": deter, **stats}
                imagined_data["context"].append(dcontext)
                imagined_data["stoch"].append(prior["stoch"])
                imagined_data["deter"].append(prior["deter"])
                imagined_data["logit"].append(prior["logit"])
                task_action, task_state = agent.task_behavior.policy(prior, task_state, dcontext=dcontext)
                prev_action = task_action["action"].sample(seed=nj.rng())
                prev_latent = prior

            # counterfactual_imagined = {**counterfactual_imagined, "context": ctx_perturbed}
            swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
            counterfactual_imagined = {key: swap(jnp.concatenate(val)) for key, val in imagined_data.items()}
            counterfactual_reconst = wm.heads["decoder"](counterfactual_imagined)
            cf_img = counterfactual_reconst["obs"].mode()
            key = f"counterfactual_dim{z_dim}_factor{midx}_noise"
            report[key] = cf_img

        report["magnitudes"] = jnp.linspace(-3, 3, 7)
        return report

    return gen_dream


def record_dream(agent, env, args, ctx_info, logdir, dream_agent_fn, ctx_id, z_dim, task):
    report = None

    def per_episode(ep):
        nonlocal agent, report, ctx_info
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)

        outdir = f"latent_traversal_ctx_id_{ctx_id}_zdim_{z_dim}"
        path = logdir / outdir
        path.mkdirs()
        original_trajectory = report["original"][0, :]
        seq_len = original_trajectory.shape[0]

        components = {
            "X Position": 0,
            "Z Position": 1,
            "X Velocity": 2,
            "Z Velocity": 3,
        }

        magnitudes = report["magnitudes"]
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(magnitudes))]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, (label, idx) in enumerate(components.items()):
            ax = axes[i]
            ax.plot(
                range(seq_len), original_trajectory[:, idx], "k-", label="Original", linewidth=2
            )
            for midx in range(7):
                color = colors[midx]
                mag_value = int(round(magnitudes[midx]))
                key = f"counterfactual_dim{z_dim}_factor{midx}_noise"
                trajectory = report[key][0, :, idx]
                ax.plot(
                    range(seq_len),
                    trajectory,
                    color=color,
                    label=f"Magnitude ≈ {mag_value}",
                    linewidth=1,
                    alpha=0.5,
                )
            ax.set_title(label, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(fontsize=10)

        fig.suptitle(
            f"Latent Traversal for Dimension {z_dim + 1}, Context: {list(ctx_info.keys())[ctx_id]}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout, leaving space for suptitle
        plt.savefig(path / f"latent_traversal_zdim{z_dim}_ctxid{ctx_id}.pdf", bbox_inches="tight")
        plt.close()

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

    parsed, other = embodied.Flags(logdir="", episodes=1, ctx_id=0, z_dim=6).parse_known()
    logdir = embodied.Path(parsed.logdir)
    ctx_id = parsed.ctx_id
    z_dim = parsed.z_dim
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
