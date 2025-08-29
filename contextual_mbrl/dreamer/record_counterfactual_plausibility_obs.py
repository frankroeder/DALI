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
    myenv = env_cls()
    # print(myenv.env.env._physics.named.data.xmat)
    # print(myenv.env.env._physics.position())
    # print(myenv.env.env._physics.velocity())
    # print("POS", myenv.env.env._physics.named.data.qpos)
    # print("VEL", myenv.env.env._physics.named.data.qvel)
    # myenv.env.env._physics.named.data.qpos["cup_x"] = 10
    # myenv.env.env._physics.named.data.qpos["cup_z"] = 20
    # myenv.env.env._physics.named.data.qpos["ball_x"] = 30
    # myenv.env.env._physics.named.data.qpos["ball_z"] = 40
    # myenv.env.env._physics.named.data.qvel["cup_x"] = 3
    # myenv.env.env._physics.named.data.qvel["cup_z"] = 4
    # myenv.env.env._physics.named.data.qvel["ball_x"] = 9
    # myenv.env.env._physics.named.data.qvel["ball_z"] = 10
    # print(myenv.env.env.task.get_observation(myenv.env.env.physics))
    context = env_cls.get_default_context()
    envs = []
    ctor = lambda: create_wrapped_carl_env(
        env_cls, contexts={0: context}, config=config
    )
    ctor = partial(embodied.Parallel, ctor, "process")
    envs = [ctor()]
    return embodied.BatchEnv(envs, parallel=True), context

def _wrap_dream_agent(agent, z_dim):
    def gen_dream(data):
        # Preprocess incoming data.
        data = agent.preprocess(data)
        wm = agent.wm

        # Use only a short sequence
        seq_len = agent.config.batch_length
        has_embed = "embed" in agent.config.ctx_encoder.inputs

        # Compute the latent context from the data via the frozen context encoder.
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


        # Compute the posterior states from the first seq_len timesteps and take the last timestep.
        posterior_states, _ = wm.rssm.observe(
            wm.encoder(data)[:, :seq_len],
            data["action"][:, :seq_len],
            data["is_first"][:, :seq_len],
            dcontext=ctx[:, :seq_len],
        )
        start = {k: v[:, 0] for k, v in posterior_states.items()}

        action = jnp.zeros_like(data["action"][:, :seq_len])

        # Generate the baseline imagined trajectory using the original actions and context.
        baseline_imagined = wm.rssm.imagine(action, start, dcontext=ctx)
        baseline_reconst = wm.heads["decoder"]({**baseline_imagined, "context": ctx})
        baseline_img = baseline_reconst["obs"].mode()

        report = {}

        # Define perturbation steps
        num_steps = 30
        sigma_z = ctx[0].std(0)[z_dim]
        deltas = jnp.linspace(sigma_z, sigma_z, num_steps)

        for idx, delta in enumerate(deltas):
            # Positive perturbation
            ctx_pos = ctx.at[:, :, z_dim].add(delta)
            counterfactual_imagined_pos = wm.rssm.imagine(action, start, dcontext=ctx_pos)
            counterfactual_reconst_pos = wm.heads["decoder"]({**counterfactual_imagined_pos, "context": ctx_pos})
            cf_img_pos = counterfactual_reconst_pos["obs"].mode()
            key = f"counterfactual_dim{z_dim}_step{idx}"
            report[key] = {
                "original": baseline_img,
                "imagined_pos": cf_img_pos,
            }
        return report
    return gen_dream

def record_dream(agent, env, args, ctx_info, logdir, dream_agent_fn, ctx_id, z_dim, task):
    report = None

    def per_episode(ep):
        nonlocal agent, report, ctx_info, env
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        report, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        report = agent._convert_mets(report, agent.train_devices)
        outdir = "obs_counterfactuals"
        path = logdir / outdir
        path.mkdirs()

        if "carl_dmc_walker" in str(logdir):
            num_rows, num_cols = 4, 6
            labels = [
                # first 14 (orientations)
                'Torso XX', 'Torso XZ',
                'Right Thigh XX', 'Right Thigh XZ',
                'Right Leg XX', 'Right Leg XZ',
                'Right Foot XX', 'Right Foot XZ',
                'Left Thigh XX', 'Left Thigh XZ',
                'Left Leg XX', 'Left Leg XZ',
                'Left Foot XX', 'Left Foot XZ',
                # center
                "Torso Z Position",
                # unsure about these velocities
                'Root Z Velocity', 'Root X Velocity', "Root Y Velocity",
                'Right HIP Velocity', 'Right Knee Velocity', 'Right Angle Velocity',
                'Left HIP Velocity', 'Left Knee Velocity', 'Left Angle Velocity',
            ]
        elif "carl_dmc_ball_in_cup" in str(logdir):
            num_rows, num_cols = 2, 4
            labels = [
                'Cup X Position', 'Cup Z Position',
                'Ball X Position', 'Ball Z Position',
                'Cup X Velocity', 'Cup Z Velocity',
                'Ball X Velocity', 'Ball Z Velocity'
            ]


        for key in report.keys():
            original = report[key]["original"][0, :]
            imagined_pos = report[key]["imagined_pos"][0, :]
            key_with_ctx = f"{key}ctxid{ctx_id}"

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows), sharex=True, constrained_layout=True)

            for dim in range(num_rows * num_cols):
                row = dim // num_cols
                col = dim % num_cols
                ax = axes[row, col]
                time_steps = range(len(original))

                ax.plot(time_steps, original[:, dim], 'o-', label='Original' if dim == 0 else "", markersize=2, linewidth=1.5, alpha=0.8)
                ax.plot(time_steps, imagined_pos[:, dim], 's-', label='Counterfactual' if dim == 0 else "", markersize=2, linewidth=1.5, alpha=0.8)

                ax.set_ylabel(labels[dim], fontsize=14)
                if row == num_rows - 1:
                    ax.set_xlabel('Time', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.set_xticks(list(range(0, len(original), 10)))

            _handles, _labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(_handles, _labels, loc='lower center', fontsize=12)
            plt.savefig(path / f"{key_with_ctx}_all_dimensions_time_series.pdf", bbox_inches='tight')
            plt.close()

            indiv_folder = path / f"{key_with_ctx}_individual_dimensions"
            indiv_folder.mkdirs()

            for dim in range(num_rows * num_cols):
                fig_dim, ax_dim = plt.subplots(figsize=(6, 4))
                time_steps = range(len(original))

                ax_dim.plot(time_steps, original[:, dim], 'o-', label='Original', markersize=2, linewidth=1.5, alpha=0.8)
                ax_dim.plot(time_steps, imagined_pos[:, dim], 's-', label='Counterfactual', markersize=2, linewidth=1.5, alpha=0.8)

                ax_dim.set_ylabel(labels[dim], fontsize=14)
                ax_dim.set_xlabel('Time', fontsize=14)
                ax_dim.grid(True, linestyle='--', alpha=0.7)
                ax_dim.tick_params(axis='both', which='major', labelsize=10)
                ax_dim.set_xticks(list(range(0, len(original), 10)))
                ax_dim.legend(fontsize=12)

                plt.savefig(indiv_folder / f"{key_with_ctx}_dim{dim}_{labels[dim].replace(' ', '_').lower()}_time_series.pdf", bbox_inches='tight')
                plt.close(fig_dim)

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
