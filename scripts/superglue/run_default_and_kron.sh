#!/usr/bin/env bash
set -euo pipefail

SUPPORTED_TASKS=(boolq cb copa axb axg wic wsc)

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/superglue/run_default_and_kron.sh [task]"
  echo "Tasks: ${SUPPORTED_TASKS[*]}"
  echo "If no task is provided, all tasks are run sequentially."
  exit 1
fi

run_task() {
  local task="$1"
  local default_script="scripts/superglue/${task}.sh"
  local kron_script="scripts/superglue/${task}_kron.sh"

  if [[ ! -f "$default_script" ]]; then
    echo "Default script not found: $default_script"
    exit 1
  fi

  if [[ ! -f "$kron_script" ]]; then
    echo "Kronecker script not found: $kron_script"
    exit 1
  fi

  echo "========================================"
  echo "Running default setting for task: $task"
  echo "Script: $default_script"
  echo "========================================"
  bash "$default_script"

  echo "========================================"
  echo "Running kronecker setting for task: $task"
  echo "Script: $kron_script"
  echo "========================================"
  bash "$kron_script"

  echo "========================================"
  echo "Completed both runs for task: $task"
  echo "========================================"
}

if [[ $# -eq 1 ]]; then
  TASK="$1"
  if [[ ! " ${SUPPORTED_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "Unsupported task: $TASK"
    echo "Supported tasks: ${SUPPORTED_TASKS[*]}"
    exit 1
  fi
  run_task "$TASK"
else
  echo "No task specified. Running all supported tasks: ${SUPPORTED_TASKS[*]}"
  for task in "${SUPPORTED_TASKS[@]}"; do
    run_task "$task"
  done
fi
