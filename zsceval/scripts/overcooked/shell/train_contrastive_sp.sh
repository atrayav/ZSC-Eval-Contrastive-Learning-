#!/bin/bash
# Train SP + contrastive partner encoder on a layout.
#
# Phase 1 (encoder trains, actor unchanged — verify encoder_loss drops):
#   bash zsceval/scripts/overcooked/shell/train_contrastive_sp.sh random3
#
# Phase 2 (encoder + actor conditioning — full method):
#   bash zsceval/scripts/overcooked/shell/train_contrastive_sp.sh random3 --condition

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zsceval
ulimit -n 65536

LAYOUT="${1:-random3}"
SEED="${SEED:-1}"
ALGO="contrastive_sp"
CONDITION_FLAG=""

if [[ "$2" == "--condition" ]]; then
  CONDITION_FLAG="--condition_actor_on_partner"
  ALGO="contrastive_sp_conditioned"
fi

if [[ "${LAYOUT}" == "random0" || "${LAYOUT}" == "random0_medium" || "${LAYOUT}" == "random1" \
   || "${LAYOUT}" == "random3" || "${LAYOUT}" == "small_corridor" || "${LAYOUT}" == "unident_s" ]]; then
  VERSION="old"
else
  VERSION="new"
fi

mkdir -p ~/ZSC-Eval/logs

echo "=== Training $ALGO on $LAYOUT seed $SEED ==="

cd ~/ZSC-Eval/zsceval/scripts/overcooked

python train/train_sp.py \
  --env_name Overcooked \
  --algorithm_name mappo \
  --experiment_name "$ALGO" \
  --layout_name "$LAYOUT" \
  --num_agents 2 \
  --seed "$SEED" \
  --n_training_threads 1 \
  --n_rollout_threads 50 \
  --dummy_batch_size 2 \
  --num_mini_batch 1 \
  --episode_length 400 \
  --num_env_steps 1e7 \
  --reward_shaping_horizon 1e8 \
  --overcooked_version "$VERSION" \
  --ppo_epoch 15 \
  --entropy_coefs 0.2 0.05 0.01 \
  --entropy_coef_horizons 0 5e6 1e7 \
  --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
  --use_recurrent_policy \
  --use_proper_time_limits \
  --save_interval 25 \
  --log_interval 10 \
  --use_eval \
  --eval_interval 20 \
  --n_eval_rollout_threads 10 \
  --use_wandb \
  --use_partner_encoder \
  --partner_emb_dim 32 \
  --encoder_context_len 20 \
  --encoder_hidden_size 64 \
  --infonce_temperature 0.1 \
  --encoder_lr 1e-3 \
  --infonce_coef 1.0 \
  $CONDITION_FLAG \
  2>&1 | tee ~/ZSC-Eval/logs/${ALGO}_${LAYOUT}_s${SEED}.log
