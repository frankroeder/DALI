#!/usr/bin/env bash

declare -a SCHEMES=(
"enc_obs_dec_obs"
"enc_img_dec_img"
"enc_obs_ctx_dec_obs_ctx"
"enc_img_ctx_dec_img_ctx"
"enc_obs_dec_obs_pgm_ctx"
"enc_img_dec_img_pgm_ctx"
"enc_obs_dec_obs_ctxencoder_transformer"
"enc_img_dec_img_ctxencoder_transformer"
"enc_obs_dec_obs_ctxencoder_transformer_grssm"
"enc_img_dec_img_ctxencoder_transformer_grssm"
)

declare -a TASKS=(
  "carl_dmc_walker"
  "carl_dmc_ball_in_cup"
)
declare -a SEEDS=("0" "42" "1337" "13" "71" "1994" "1997" "908" "2102" "3")
declare -a CONTEXTS=("single_0" "single_1" "double_box")

cd "$(git rev-parse --show-toplevel || echo .)"
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for CROSSMODAL in "True False"; do
      for CONTEXT in "${CONTEXTS[@]}"; do
        for SCHEME in "${SCHEMES[@]}"; do

          if [ "$TASK" == "carl_dmc_walker" ]; then
            STEPS=500000
          elif [ "$TASK" == "carl_dmc_ball_in_cup" ]; then
            STEPS=200000
          fi

          CONTEXT_MOD=$CONTEXT
          SCHEME_MOD=$SCHEME

          if [ "$SCHEME" == "enc_obs_dec_obs_default" ]; then
            # exit if context is not single_0 as we only want to run the default scheme once
            if [ "$CONTEXT" != "single_0" ]; then
              continue
            fi
            SCHEME_MOD="enc_obs_dec_obs"
            CONTEXT_MOD="default"
          elif [ "$SCHEME" == "enc_img_dec_img_default" ]; then
            if [ "$CONTEXT" != "single_0" ]; then
              continue
            fi
            SCHEME_MOD="enc_img_dec_img"
            CONTEXT_MOD="default"
          fi

          GROUP_NAME="${TASK}_${CONTEXT_MOD}_${SCHEME_MOD}_normalized"
          echo "GROUPNAME IS $GROUP_NAME"

          if [ "$CROSSMODAL" = "True" ]; then
            LOGDIR="logs_dali_crossmodal/$GROUP_NAME/$SEED"
          else
            LOGDIR="logs_dali/$GROUP_NAME/$SEED"
          fi
          echo "LOGDIR IS $LOGDIR"

        # train
        uv run -m contextual_mbrl.dreamer.train --configs carl $SCHEME_MOD  --jax.prealloc False \
          --task $TASK --env.carl.context $CONTEXT_MOD --seed $SEED --ctx_encoder.crossmodal $CROSSMODAL \
          --logdir "$LOGDIR" --wandb.project '' --wandb.group $GROUP_NAME --run.steps $STEPS --jax.platform gpu && \
          uv run -m contextual_mbrl.dreamer.eval --logdir "$LOGDIR"
        done
      done
    done
  done
done
