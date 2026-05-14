#!/bin/bash
#SBATCH --job-name="dlra"
#SBATCH --partition="rocky"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-23:59
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --constraint=type_e

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-roberta-base}"
TRACKING_BACKEND="${TRACKING_BACKEND:-comet}"
SEED="${SEED:-1234}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-30}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
POWER_ITERATIONS="${POWER_ITERATIONS:-3}"
FACTORS_RANK="${FACTORS_RANK:-32}"
DLRA_RANK="${DLRA_RANK:-32}"
LOW_RANK_SCALE="${LOW_RANK_SCALE:-4}"
SUBSPACE_UPDATE_INTERVAL="${SUBSPACE_UPDATE_INTERVAL:-100}"
ST_INIT_STEP_SIZE="${ST_INIT_STEP_SIZE:-50.0}"
TRUNCATION_EPS="${TRUNCATION_EPS:-1e-3}"
RANGEFINDER_TAU="${RANGEFINDER_TAU:-1e-3}"
RANGEFINDER_BETA="${RANGEFINDER_BETA:-1e-5}"
DLRA_UPDATE_BETA="${DLRA_UPDATE_BETA:-0.99}"
USE_LORA_ALL_MODULES="${USE_LORA_ALL_MODULES:-1}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-superglue-copa-base}"

TARGET_MODULE_ARGS=()
if [[ "${USE_LORA_ALL_MODULES}" == "1" ]]; then
  TARGET_MODULE_ARGS=(--lora_all_modules)
fi

COMMON_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --task_name copa
  --max_length "${MAX_LENGTH}"
  --seed "${SEED}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --tracking_backend "${TRACKING_BACKEND}"
)

export HF_DEBUG=1 
export HF_HUB_VERBOSITY=debug
export HF_HUB_DISABLE_XET=1

run_variant() {
  local label="$1"
  shift

  echo "========================================"
  echo "Running ${label} on SuperGLUE/copa"
  echo "========================================"

  python run_super_glue.py "${COMMON_ARGS[@]}" "$@"
}

# LOW_RANK_COMMON_ARGS=(
#   --enable_low_rank
#   "${TARGET_MODULE_ARGS[@]}"
#   --lora_r "${DLRA_RANK}"
#   --low_rank_scale "${LOW_RANK_SCALE}"
#   --subspace_update_interval "${SUBSPACE_UPDATE_INTERVAL}"
#   --weight_decay "${WEIGHT_DECAY}"
# )

# run_variant "GaLore" \
#   "${LOW_RANK_COMMON_ARGS[@]}" \
#   --subspace_update_method galore \
#   --kronecker_mode none \
#   --run_name "${RUN_NAME_PREFIX}-galore"

# run_variant "SubTrack++" \
#   "${LOW_RANK_COMMON_ARGS[@]}" \
#   --subspace_update_method subtrack \
#   --st_init_step_size "${ST_INIT_STEP_SIZE}" \
#   --adaptive_optimizer \
#   --recovery_scaling \
#   --kronecker_mode none \
#   --run_name "${RUN_NAME_PREFIX}-subtrackpp"

# run_variant "DyKAF" \
#   "${TARGET_MODULE_ARGS[@]}" \
#   --enable_dykaf \
#   --power_iterations "${POWER_ITERATIONS}" \
#   --weight_decay "${WEIGHT_DECAY}" \
#   --run_name "${RUN_NAME_PREFIX}-dykaf"


run_variant "LowRankDyKAF PSI" \
  "${TARGET_MODULE_ARGS[@]}" \
  --enable_dykaf \
  --low_rank_factors \
  --factors_rank "${FACTORS_RANK}" \
  --low_rank_proj psi \
  --power_iterations "${POWER_ITERATIONS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --run_name "${RUN_NAME_PREFIX}-low-rank-dykaf-psi"


run_variant "DLRAdamW PSI" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --dlra_projection dlra \
  --dlra_update_mode add \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlradamw-psi"

  run_variant "DLRAdamW PSI" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --dlra_projection dlra \
  --dlra_update_mode ema \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlradamw-ema-psi"


run_variant "LowRankDyKAF adaptive-rand" \
  "${TARGET_MODULE_ARGS[@]}" \
  --enable_dykaf \
  --low_rank_factors \
  --factors_rank "${FACTORS_RANK}" \
  --low_rank_proj rand \
  --truncation_eps "${TRUNCATION_EPS}" \
  --rangefinder_tau "${RANGEFINDER_TAU}" \
  --rangefinder_beta "${RANGEFINDER_BETA}" \
  --power_iterations "${POWER_ITERATIONS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --run_name "${RUN_NAME_PREFIX}-low-rank-dykaf-adaptive-rand"

run_variant "DLRAdamW adaptive-rand" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --dlra_projection rand_svd \
  --adaptive_rangefinder \
  --truncation_eps "${TRUNCATION_EPS}" \
  --rangefinder_tau "${RANGEFINDER_TAU}" \
  --rangefinder_beta "${RANGEFINDER_BETA}" \
  --dlra_update_mode add \
  --power_iterations "${POWER_ITERATIONS}" \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlradamw-adaptive-rand"


run_variant "DLRAdamW adaptive-rand" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --dlra_projection dlra \
  --dlra_update_mode ema \
  --dlra_update_beta "${DLRA_UPDATE_BETA}" \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlradamw-adaptive-rand-nystrom"


run_variant "DLRAdamW adaptive-rand" \
  "${LOW_RANK_COMMON_ARGS[@]}" \
  --low_rank_method dlr \
  --dlra_projection rand_nystrom \
  --adaptive_rangefinder \
  --truncation_eps "${TRUNCATION_EPS}" \
  --rangefinder_tau "${RANGEFINDER_TAU}" \
  --rangefinder_beta "${RANGEFINDER_BETA}" \
  --dlra_update_mode ema \
  --dlra_update_beta "${DLRA_UPDATE_BETA}" \
  --power_iterations "${POWER_ITERATIONS}" \
  --kronecker_mode none \
  --run_name "${RUN_NAME_PREFIX}-dlradamw-ema-adaptive-rand-nystrom"


echo "========================================"
echo "Completed DyKAF/DLRA COPA suite"
echo "========================================"
