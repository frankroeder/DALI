#!/usr/bin/env bash

# Define tasks, seeds, schemes, and contexts
declare -a TASKS=("carl_dmc_ball_in_cup")
declare -a SEEDS=("0" "42" "1337" "13" "71")
declare -a SCHEMES=("enc_obs_dec_obs")
declare -a CONTEXTS=(
    "specific_0-0.98" "specific_0-2.45" "specific_0-3.92"
    "specific_0-4.9" "specific_0-9.81" "specific_0-14.7"
    "specific_0-15.68" "specific_0-17.64" "specific_0-19.6"
    "specific_1-0.03" "specific_1-0.09" "specific_1-0.15"
    "specific_1-0.225" "specific_1-0.3" "specific_1-0.45"
    "specific_1-0.54" "specific_1-0.6" "specific_0-2.45_1-0.09"
    "specific_0-17.64_1-0.09" "specific_0-17.64_1-0.54"
    "specific_0-2.45_1-0.54"
)

cd "$(git rev-parse --show-toplevel || echo .)"

# Loop through combinations of tasks, seeds, schemes, and contexts
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for SCHEME in "${SCHEMES[@]}"; do
      for CONTEXT in "${CONTEXTS[@]}"; do

        GROUP_NAME="${TASK}_${CONTEXT}_${SCHEME}"
        echo "GROUPNAME IS $GROUP_NAME"

        LOGDIR="logs_dali/specific/$GROUP_NAME/$SEED"
        echo "LOGDIR IS $LOGDIR"

        # Train and evaluate
        uv run -m contextual_mbrl.dreamer.train --configs carl $SCHEME \
          --task $TASK --env.carl.context $CONTEXT --seed $SEED \
          --logdir "$LOGDIR" --wandb.project '' --wandb.group $GROUP_NAME --run.steps 200000 --jax.platform gpu && \
          uv run -m contextual_mbrl.dreamer.eval --logdir "$LOGDIR" && \
					uv run -m contextual_mbrl.dreamer.eval --logdir "$LOGDIR" --random_policy True
      done
    done
  done
done
