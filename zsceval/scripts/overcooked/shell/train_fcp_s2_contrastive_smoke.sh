#!/bin/bash
# Smoke test for the staged contrastive setup: FCP Stage 2 with the adaptive
# agent conditioned on the FROZEN offline pre-trained partner encoder.
# Tiny run (2 updates, 12-partner smoke population) - checks wiring only.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate zsceval
cd ~/ZSC-Eval/zsceval/scripts/overcooked
export POLICY_POOL=../../policy_pool
mkdir -p ~/ZSC-Eval/logs
ulimit -n 65536 2>/dev/null || ulimit -n 4096

python train/train_adaptive.py \
  --env_name Overcooked \
  --algorithm_name adaptive \
  --experiment_name "fcp-S2-contrastive-smoke" \
  --layout_name random3 \
  --num_agents 2 \
  --seed 1 \
  --n_training_threads 1 \
  --num_mini_batch 1 \
  --episode_length 400 \
  --num_env_steps 9600 \
  --reward_shaping_horizon 9600 \
  --overcooked_version old \
  --n_rollout_threads 12 \
  --dummy_batch_size 2 \
  --ppo_epoch 5 \
  --entropy_coefs 0.2 0.05 0.01 \
  --entropy_coef_horizons 0 5000 10000 \
  --stage 2 \
  --save_interval 25 \
  --log_interval 1 \
  --use_eval \
  --eval_interval 20 \
  --n_eval_rollout_threads 24 \
  --eval_episodes 5 \
  --population_yaml_path ../../policy_pool/random3/fcp/s2/train-smoke.yml \
  --population_size 12 \
  --adaptive_agent_name fcp_adaptive \
  --use_agent_policy_id \
  --use_proper_time_limits \
  --use_wandb \
  --use_partner_encoder \
  --condition_actor_on_partner \
  --partner_emb_dim 32 \
  --encoder_context_len 20 \
  --encoder_hidden_size 64 \
  --pretrained_encoder_path ~/ZSC-Eval/data/partner_windows/encoder_random3.pt \
  2>&1 | tee ~/ZSC-Eval/logs/fcp_s2_contrastive_smoke_random3.log
