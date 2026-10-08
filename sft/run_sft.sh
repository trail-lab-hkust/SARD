#!/usr/bin/env bash
set -euo pipefail

# Run this script from a LLaMA-Factory checkout, or set LLAMA_FACTORY_DIR.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/external/LLaMA-Factory}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/qwen_full_sft.yaml}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"

mkdir -p "${LOG_DIR}"
cd "${LLAMA_FACTORY_DIR}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-SARD-SFT}"

FORCE_TORCHRUN=1 llamafactory-cli train "${CONFIG_PATH}" "$@" 2>&1 | tee "${LOG_DIR}/sft.log"
