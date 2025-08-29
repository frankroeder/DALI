import sys
import csv
import logging
import os
import re
import warnings
import pickle

import dreamerv3
import jsonlines
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied

from contextual_mbrl.dreamer.envs import gen_carl_val_envs

logging.captureWarnings(True)
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"  # use EGL instead of GLFW to render MuJoCo


def eval(policy, env, args, episodes=10):
    report = []
    lengths = []
    rewards = []

    def per_episode(ep):
        report.append({
            "obs": ep["obs"],
            "action": ep["action"],
            "context": ep["context"]
        })
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        lengths.append(length)
        rewards.append(score)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))

    print("Start evaluation loop.")
    driver(policy, episodes=episodes)
    metrics = {
        "length": np.mean(lengths).astype(float),
        "length_std": np.std(lengths).astype(float),
        "length_min": np.min(lengths).astype(float),
        "length_max": np.max(lengths).astype(float),
        "return": np.mean(rewards).astype(float),
        "return_std": np.std(rewards).astype(float),
        "return_min": np.min(rewards).astype(float),
        "return_max": np.max(rewards).astype(float),
        "returns": rewards,
        "lengths": lengths,
    }

    return metrics, report


def create_random_policy(act_space):
    def policy(*args):
        bs = args[0]["is_first"].shape[0]
        actions_space = act_space["action"]
        actions = np.stack([actions_space.sample() for _ in range(bs)])
        return {"action": actions}, None

    return policy


def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    warnings.filterwarnings("once", ".*If you want to use these environments.*")
    warnings.filterwarnings("module", "carl.*")

    # create argparse with logdir
    parsed, other = embodied.Flags(logdir="", random_policy=False).parse_known()
    logdir = embodied.Path(parsed.logdir)
    is_random_policy = parsed.random_policy
    # load the config from the logdir
    config = yaml.YAML(typ="safe").load((logdir / "config.yaml").read())
    config = embodied.Config(config)
    # parse the overrides for eval
    config = embodied.Flags(config).parse(other)

    checkpoint = logdir / "checkpoint.ckpt"
    assert checkpoint.exists(), checkpoint
    config = config.update({"run.from_checkpoint": str(checkpoint)})

    # Just load the step counter from the checkpoint, as there is
    # a circular dependency to load the agent.
    step = 0
    if not is_random_policy:
        ckpt = embodied.Checkpoint()
        ckpt.step = embodied.Counter()
        ckpt.load(checkpoint, keys=["step"])
        step = ckpt._values["step"]

    policy = None
    returns = []
    lengths = []
    ctx2latent = []
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    from contextual_mbrl.dreamer.envs import (
        _TASK2CONTEXTS,
    )
    suite, task = config.task.split("_", 1)
    ctx_0 = _TASK2CONTEXTS[task][0]["context"]
    ctx_1 = _TASK2CONTEXTS[task][1]["context"]

    for env, ctx_info in gen_carl_val_envs(config):

        if policy is None:
            if is_random_policy:
                policy = create_random_policy(env.act_space)
            else:
                agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
                checkpoint = embodied.Checkpoint()
                checkpoint.agent = agent
                checkpoint.load(args.from_checkpoint, keys=["agent"])
                policy = lambda *args: agent.policy(*args, mode="eval")
        metrics, logs = eval(policy, env, args, episodes=50)
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
        returns.extend(metrics["returns"])
        lengths.extend(metrics["lengths"])
        metrics["ctx"] = {**ctx_info}
        metrics["aggregated_context_metric"] = False
        metrics["checkpoint_step"] = int(step)

        # Write metrics to eval.jsonl
        log_file = logdir / "eval.jsonl"
        if is_random_policy:
            log_file = logdir / "eval_random_policy.jsonl"

        with jsonlines.open(log_file, mode="a") as writer:
            writer.write(metrics)

    with (logdir / f"{suite}_{task}_random_policy_data.pkl").open("wb") as f:
        pickle.dump(ctx2latent, f)


if __name__ == "__main__":
    main()
