#!/bin/bash
# Full-scale FCP Stage 2 with the adaptive agent conditioned on the FROZEN
# offline pre-trained partner encoder (staged contrastive setup).
#
# Usage:
#   bash train_fcp_s2_contrastive.sh <layout> <population_size> [mode]
#     layout           e.g. random3
#     population_size  12 | 24 | 36 (population yamls from prep/gen_S2_yml.py)
#     mode             "contrastive" (default) or "baseline"
#                      baseline = identical run WITHOUT the encoder — the
#                      same-codebase FCP control for the BR-Prox comparison
#
# Env overrides: SEED_BEGIN/SEED_MAX (default 1..5), N_THREADS (default 100),
#   ENCODER_CKPT (default data/partner_windows/encoder_<layout>.pt — the 3k-step
#   checkpoint; the 10k one overfits, see RESEARCH_LOG 2026-07-12).
#
# Requires (regenerate on a fresh machine, both are gitignored):
#   - population yamls: cd zsceval/scripts && python prep/gen_S2_yml.py <layout> fcp
#   - encoder ckpt: collect_partner_windows.py + pretrain_partner_encoder.py
env="Overcooked"

layout=$1
population_size=$2
mode=${3:-contrastive}
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

if [[ ${population_size} == 12 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e7 5e7"
    reward_shaping_horizon="5e7"
    num_env_steps="5e7"
elif [[ ${population_size} == 24 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 4e7 8e7"
    reward_shaping_horizon="8e7"
    num_env_steps="8e7"
elif [[ ${population_size} == 36 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 5e7 1e8"
    reward_shaping_horizon="1e8"
    num_env_steps="1e8"
else
    echo "population_size must be 12, 24 or 36" >&2; exit 1
fi
pop="sp"

encoder_ckpt=${ENCODER_CKPT:-~/ZSC-Eval/data/partner_windows/encoder_${layout}.pt}
encoder_flags=""
if [[ "${mode}" == "contrastive" ]]; then
    [[ -f ${encoder_ckpt/#\~/$HOME} ]] || { echo "missing encoder ckpt ${encoder_ckpt}" >&2; exit 1; }
    encoder_flags="--use_partner_encoder --condition_actor_on_partner \
        --partner_emb_dim 32 --encoder_context_len 20 --encoder_hidden_size 64 \
        --pretrained_encoder_path ${encoder_ckpt}"
fi

num_agents=2
algo="adaptive"
exp="fcp-S2-${mode}-s${population_size}"
seed_begin=${SEED_BEGIN:-1}
seed_max=${SEED_MAX:-5}
n_training_threads=${N_THREADS:-100}

source ~/miniconda3/etc/profile.d/conda.sh && conda activate zsceval
cd "$(dirname "$0")/.."
path=../../policy_pool
export POLICY_POOL=${path}
mkdir -p ~/ZSC-Eval/logs

ulimit -n 65536 2>/dev/null || ulimit -n 4096

echo "env ${env}, layout ${layout}, mode ${mode}, exp ${exp}, seeds ${seed_begin}..${seed_max}, steps ${num_env_steps}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --n_rollout_threads ${n_training_threads} --dummy_batch_size 2 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --stage 2 \
    --save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 5 \
    --population_yaml_path ${path}/${layout}/fcp/s2/train-s${population_size}-${pop}-${seed}.yml \
    --population_size ${population_size} --adaptive_agent_name fcp_adaptive --use_agent_policy_id \
    --use_proper_time_limits \
    --use_wandb \
    ${encoder_flags} \
    2>&1 | tee ~/ZSC-Eval/logs/${exp}_${layout}_s${seed}.log
done
