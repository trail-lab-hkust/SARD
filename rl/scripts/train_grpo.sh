#!/usr/bin/env bash
set -euo pipefail
set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-SARD-RL}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VERL_DIR="${VERL_DIR:-${PROJECT_ROOT}/external/verl}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RUN_DIR="${PROJECT_ROOT}/rl"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
cd "${VERL_DIR}"

MODEL_PATH=${MODEL_PATH:-/path/to/sft_checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sard_v2_grpo_mixed_132}
GUIDED_BATCH_DUMP_ENABLE=${GUIDED_BATCH_DUMP_ENABLE:-True}
GUIDED_BATCH_DUMP_EVERY_N_STEPS=${GUIDED_BATCH_DUMP_EVERY_N_STEPS:-1}
GUIDED_BATCH_DUMP_DIR=${GUIDED_BATCH_DUMP_DIR:-${OUTPUT_DIR}/guided_question_dumps}
REPAIR_DUMP_RECORDS=${GUIDED_REPAIR_DUMP_RECORDS:-True}
REPAIR_DUMP_DIR=${GUIDED_REPAIR_DUMP_DIR:-${OUTPUT_DIR}/guided_repair_dumps}
REPAIR_MAX_ROUNDS=${GUIDED_REPAIR_MAX_ROUNDS:-3}
REPAIR_CONCURRENCY=${GUIDED_REPAIR_LLM_CONCURRENCY:-16}
REPAIR_TIMEOUT_SECONDS=${GUIDED_REPAIR_LLM_TIMEOUT_SECONDS:-60}
REPAIR_MAX_RETRIES=${GUIDED_REPAIR_LLM_MAX_RETRIES:-3}
GUIDED_REPAIR_LLM_BASE_URL="${GUIDED_REPAIR_LLM_BASE_URL:-https://api.example.com/v1}"
GUIDED_REPAIR_LLM_MODEL="${GUIDED_REPAIR_LLM_MODEL:-deepseek-v3.2-exp}"

log_stamp=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/train_grpo_sard_grpo_${log_stamp}.log"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train_rl_sard_v2_guided.parquet" \
    data.val_files="${DATA_DIR}/val_rl_sard_v2_guided.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=15360 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    ++algorithm.guided_grpo.enable=False \
    ++algorithm.guided_grpo.direct_teacher_fallback=False \
    ++algorithm.guided_grpo.routing_metric=ndcg@10 \
    ++algorithm.guided_grpo.success_threshold=0.9 \
    ++algorithm.guided_grpo.off_policy_gamma=0.1 \
    ++algorithm.guided_grpo.train_auxiliary_groups=False \
    ++algorithm.guided_grpo.auxiliary_group_loss_weight=0.1 \
    ++algorithm.guided_grpo.teacher_fallback_aux_group=none \
    ++algorithm.guided_grpo.batch_dump.enable="${GUIDED_BATCH_DUMP_ENABLE}" \
    ++algorithm.guided_grpo.batch_dump.every_n_steps="${GUIDED_BATCH_DUMP_EVERY_N_STEPS}" \
    ++algorithm.guided_grpo.batch_dump.dump_dir="${GUIDED_BATCH_DUMP_DIR}" \
    ++algorithm.guided_grpo.repair.enable=False \
    ++algorithm.guided_grpo.repair.max_rounds="${REPAIR_MAX_ROUNDS}" \
    ++algorithm.guided_grpo.repair.dump_records="${REPAIR_DUMP_RECORDS}" \
    ++algorithm.guided_grpo.repair.dump_dir="${REPAIR_DUMP_DIR}" \
    ++algorithm.guided_grpo.repair.llm.provider=openai_compatible \
    ++algorithm.guided_grpo.repair.llm.base_url="${GUIDED_REPAIR_LLM_BASE_URL}" \
    ++algorithm.guided_grpo.repair.llm.model="${GUIDED_REPAIR_LLM_MODEL}" \
    ++algorithm.guided_grpo.repair.llm.api_key_env=GUIDED_REPAIR_LLM_API_KEY \
    ++algorithm.guided_grpo.repair.llm.timeout_seconds="${REPAIR_TIMEOUT_SECONDS}" \
    ++algorithm.guided_grpo.repair.llm.max_retries="${REPAIR_MAX_RETRIES}" \
    ++algorithm.guided_grpo.repair.llm.concurrency="${REPAIR_CONCURRENCY}" \
    ++algorithm.guided_grpo.repair.llm.temperature=0.2 \
    ++algorithm.guided_grpo.repair.llm.max_tokens=4096 \
    ++algorithm.guided_grpo.repair.llm.enable_thinking=False \
    ++data.strategy_prompt_key=__unused_strategy_prompt_for_grpo__ \
    ++data.teacher_response_key=teacher_response \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='sard_rl' \
    trainer.experiment_name='grpo_sard_grpo' \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee -a "${log_file}"
