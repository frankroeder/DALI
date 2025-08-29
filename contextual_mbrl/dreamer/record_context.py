import logging
import os, sys
import pickle
import warnings

import dreamerv3
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied
from dreamerv3 import ninjax as nj

from contextual_mbrl.dreamer.envs import (
    _TASK2CONTEXTS,
)
from contextual_mbrl.dreamer.envs import gen_carl_val_envs

logging.captureWarnings(True)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def _wrap_dream_agent(agent):
    def gen_dream(data):
        # Preprocess the batch of episode data.
        data = agent.preprocess(data)
        wm = agent.wm
        ctx = None
        embed = wm.encoder(data)

        if wm.use_ctx_encoder:
            # Extract dimensions for clarity
            batch_size = int(data["obs"].shape[0])
            num_steps = int(data["obs"].shape[1])
            context_len = int(agent.config["batch_length"])
            obs_dim = int(data["obs"].shape[-1])
            action_dim = int(data["action"].shape[-1])

            # Determine context dimension and output shape with a test run
            dummy_window = {
                "obs": data["obs"][:, :context_len],
                "action": data["action"][:, :context_len],
                "embed": embed[:, :context_len]
            }

            # Run a test to determine output shape
            dummy_ctx = wm.ctx_encoder(dummy_window)

            # Determine if ctx_encoder returns per-step or single context vector
            if len(dummy_ctx.shape) == 3:  # (batch, time, ctx_dim)
                context_dim = dummy_ctx.shape[-1]
                returns_sequence = True
            else:  # (batch, ctx_dim)
                context_dim = dummy_ctx.shape[-1]
                returns_sequence = False

            # Create padded data for sliding window
            pad_width = context_len - 1
            padded_obs = jnp.pad(
                data["obs"],
                ((0, 0), (pad_width, 0), (0, 0)),
                mode='edge'  # Use edge padding instead of zeros for more realistic context
            )
            padded_action = jnp.pad(
                data["action"],
                ((0, 0), (pad_width, 0), (0, 0)),
                mode='edge'
            )
            embed_dim = embed.shape[-1]
            padded_embed = jnp.pad(
                embed,
                ((0, 0), (pad_width, 0), (0, 0)),
                mode='edge'
            )

            # Preallocate results array
            ctx_array = jnp.zeros((batch_size, num_steps, context_dim))

            # Define processing function for each timestep
            def process_timestep(i, ctx_array):
                # Create mask where earlier timesteps have value 1
                # For timestep i, we want to include data from max(0, i-context_len+1) to i
                valid_length = jnp.minimum(i + 1, context_len)
                # mask = jnp.arange(context_len) >= (context_len - valid_length)

                # Extract context window ending at current timestep
                start_idx = jnp.maximum(0, i - context_len + 1)
                window_obs = jax.lax.dynamic_slice(
                    padded_obs,
                    (0, i - valid_length + 1, 0),
                    (batch_size, context_len, obs_dim)
                )
                window_action = jax.lax.dynamic_slice(
                    padded_action,
                    (0, i - valid_length + 1, 0),
                    (batch_size, context_len, action_dim)
                )
                window_embed = jax.lax.dynamic_slice(
                    padded_embed,
                    (0, i - valid_length + 1, 0),
                    (batch_size, context_len, embed_dim)
                )

                # Prepare window data
                window_data = {
                    "obs": window_obs,
                    "action": window_action,
                    "embed" : window_embed
                }

                # Get context for this window
                timestep_ctx = wm.ctx_encoder(window_data)

                # Extract the appropriate context vector
                if returns_sequence:
                    # If encoder returns sequence, take last valid step
                    ctx_for_step = timestep_ctx[:, -1]
                else:
                    # Already a single context vector per batch
                    ctx_for_step = timestep_ctx

                # Update the context array
                ctx_array = ctx_array.at[:, i].set(ctx_for_step)
                return ctx_array

            # Process all timesteps
            ctx_array = jax.lax.fori_loop(0, num_steps, process_timestep, ctx_array)
            ctx = ctx_array
        elif wm.rssm._add_dcontext:
            ctx = data["context"]

        report = {}
        ep_len = data["is_first"].shape[1]
        if ep_len == 0:
            return report

        # Run the RSSM observer on the entire episode.
        posterior_states, prior_states = wm.rssm.observe(
            wm.encoder(data),
            data["action"],
            data["is_first"],
            dcontext=ctx,
        )
        # Limit imagined latents
        # start = {k: v[:, 0] for k, v in posterior_states.items()}
        # imagined_states = wm.rssm.imagine(
        #     data["action"],
        #     start,
        #     dcontext=ctx,
        # )

        # Collect the full episode data.
        report["obs"] = data["obs"][0]  # Full observation sequence.
        report["real_context"] = data["context"][
            0
        ]  # The raw context signal from the env.
        if ctx is not None:
            report["context_representation"] = ctx[
                0
            ]  # The context encoder's representation.
        report["image"] = data["image"][0]  # Full image sequence.
        report["reward"] = data["reward"][0]
        report["reset"] = data["reset"][0]
        report["action"] = data["action"][0]
        report["log_entropy"] = data["log_entropy"][0]
        report["embed"] = embed[0]

        # Combine the latent states from the observed (posterior) trajectory.
        deter = posterior_states["deter"][0]  # Deterministic part.
        stoch = posterior_states["stoch"][0]  # Stochastic part.
        stoch = jnp.reshape(stoch, (stoch.shape[0], -1))
        report["posterior"] = jnp.concatenate([deter, stoch], axis=1)

        prior_deter = prior_states["deter"][0]  # Deterministic part.
        _stoch = prior_states["stoch"][0]  # Stochastic part.
        prior_stoch = jnp.reshape(_stoch, (_stoch.shape[0], -1))
        report["prior"] = jnp.concatenate([prior_deter, prior_stoch], axis=1)

        # Combine the latent states from the imagined rollout.
        # deter_imag = imagined_states["deter"][0]
        # stoch_imag = imagined_states["stoch"][0]
        # stoch_imag = jnp.reshape(stoch_imag, (stoch_imag.shape[0], -1))
        # report["imagined"] = jnp.concatenate([deter_imag, stoch_imag], axis=1)

        return report

    return gen_dream


def collect_ctx_representations(agent, env, args, dream_agent_fn, episodes):
    report = []

    def per_episode(ep):
        nonlocal agent, report
        # Stack episode data into a batch.
        batch = {k: np.stack([v], 0) for k, v in ep.items()}
        jax_batch = agent._convert_inps(batch, agent.train_devices)
        rng = agent._next_rngs(agent.train_devices)
        r, _ = dream_agent_fn(agent.varibs, rng, jax_batch)
        if r:
            r = agent._convert_mets(r, agent.train_devices)
            report.append(r)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=episodes)

    return report


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # Create argparse with logdir and episodes.
    parsed, other = embodied.Flags(logdir="", episodes=1).parse_known()
    logdir = embodied.Path(parsed.logdir)

    # Load the config from the logdir.
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config["envs"]["amount"] = 2
    config = embodied.Config(config)
    # Parse any command-line overrides for evaluation.
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Load only the step counter from the checkpoint to avoid circular dependencies.
    ckpt = embodied.Checkpoint()
    ckpt.step = embodied.Counter()
    ckpt.load(checkpoint, keys=["step"])
    step = ckpt._values["step"]

    dream_agent_fn = None
    agent = None
    ctx2latent = []
    suite, task = config.task.split("_", 1)
    ctx_0 = _TASK2CONTEXTS[task][0]["context"]
    ctx_1 = _TASK2CONTEXTS[task][1]["context"]

    for env, ctx_info in gen_carl_val_envs(config):
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            dream_agent_fn = nj.pure(_wrap_dream_agent(agent.agent))
            dream_agent_fn = nj.jit(dream_agent_fn, device=agent.train_devices[0])
        args = embodied.Config(
            **config.run,
            logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length,
        )
        logs = collect_ctx_representations(
            agent,
            env,
            args,
            dream_agent_fn,
            parsed.episodes,
        )
        ctx2latent.append(
            {
                "context": {
                    ctx_0: ctx_info["context"][ctx_0],
                    ctx_1: ctx_info["context"][ctx_1],
                },
                "context_order": [ctx_0, ctx_1],
                "episodes": logs,
            }
        )
        env.close()

    with (logdir / f"{suite}_{task}_ctx_represenations_windows.pkl").open("wb") as f:
        pickle.dump(ctx2latent, f)


if __name__ == "__main__":
    main()
