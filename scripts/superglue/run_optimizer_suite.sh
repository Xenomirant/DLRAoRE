#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-copa}"

SUPPORTED_TASKS=(boolq cb copa axb axg wic wsc)

if [[ ! " ${SUPPORTED_TASKS[*]} " =~ " ${TASK} " ]]; then
  echo "Unsupported task: ${TASK}"
  echo "Supported tasks: ${SUPPORTED_TASKS[*]}"
  exit 1
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-roberta-large}"
TRACKING_BACKEND="${TRACKING_BACKEND:-none}"
SEED="${SEED:-1234}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-30}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
LORA_R="${LORA_R:-8}"
LOW_RANK_SCALE="${LOW_RANK_SCALE:-4}"
SUBSPACE_UPDATE_INTERVAL="${SUBSPACE_UPDATE_INTERVAL:-100}"
ST_INIT_STEP_SIZE="${ST_INIT_STEP_SIZE:-50.0}"
FACTORS_RANK="${FACTORS_RANK:-16}"
FACTORS_OVERSAMPLE="${FACTORS_OVERSAMPLE:-5}"
POWER_ITERATIONS="${POWER_ITERATIONS:-3}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-superglue-${TASK}}"

COMMON_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --task_name "${TASK}"
  --max_length "${MAX_LENGTH}"
  --seed "${SEED}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --tracking_backend "${TRACKING_BACKEND}"
)

LOW_RANK_COMMON_ARGS=(
  --enable_low_rank
  --lora_all_modules
  --lora_r "${LORA_R}"
  --low_rank_scale "${LOW_RANK_SCALE}"
  --subspace_update_interval "${SUBSPACE_UPDATE_INTERVAL}"
)

run_variant() {
  local label="$1"
  shift

  echo "========================================"
  echo "Running ${label} on SuperGLUE/${TASK}"
  echo "========================================"

  python run_super_glue.py "${COMMON_ARGS[@]}" "$@"
}

run_variant "SubTrack++" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --subspace_update_method subtrack \
  --st_init_step_size "${ST_INIT_STEP_SIZE}" \
  --adaptive_optimizer \
  --recovery_scaling \
  --kronecker_mode auto \
  --run_name "${RUN_NAME_PREFIX}-subtrackpp"

run_variant "DyKAF" \
  --enable_dykaf \
  --power_iterations "${POWER_ITERATIONS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --run_name "${RUN_NAME_PREFIX}-dykaf"

run_variant "LowRankDyKAF" \
  --enable_dykaf \
  --low_rank_factors \
  --factors_rank "${FACTORS_RANK}" \
  --factors_oversample "${FACTORS_OVERSAMPLE}" \
  --low_rank_proj psi \
  --power_iterations "${POWER_ITERATIONS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --run_name "${RUN_NAME_PREFIX}-low-rank-dykaf"

run_variant "LowKronRankDyKAF" \
  --enable_dykaf \
  --low_rank_factors \
  --factors_rank "${FACTORS_RANK}" \
  --factors_oversample "${FACTORS_OVERSAMPLE}" \
  --low_rank_proj psi \
  --power_iterations "${POWER_ITERATIONS}" \
  --factor_kronecker_mode auto \
  --weight_decay "${WEIGHT_DECAY}" \
  --run_name "${RUN_NAME_PREFIX}-low-kron-rank-dykaf"

run_variant "DLRAdamW" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --weight_decay "${WEIGHT_DECAY}" \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlr-adamw"

echo "========================================"
echo "Completed optimizer suite for SuperGLUE/${TASK}"
echo "========================================"
