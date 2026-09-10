#!/usr/bin/env bash
# Qwen3-1.7B ReasoningQAT — paper-faithful Stage 1+2+3.
#
# Differs from scripts/run_qwen3_1.7b.sh (Exp #1 offline JSD+plain CE):
#   Stage 2 uses teacher-guided reward rectification (--use_teacher_weight)
#   + forward KL(teacher||student) on top-20, cosine LR schedule.
#   Outputs go under *-reasoningqat so Exp #1 is not overwritten.
#
# Paper objective:
#   L = 0.2 * CE_gold * sg(pi_T(y*)) + 1.0 * KL(pi_T || pi_S)
#
# Usage:
#   bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --stage 2 --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_qwen3_1.7b_reasoningqat.sh --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WBITS=3
GROUP_SIZE=128
STAGE="all"
TRAIN_EMB=0
SKIP_EXISTING=0
EXP_NAME_STEM="Qwen3-1.7B"
MODEL_CLI=0
TEACHER_CLI=0

MODEL="${MODEL_PATH:-/zju_0038/zq/models/Qwen3-1.7B}"
TEACHER="${TEACHER_MODEL:-${MODEL}}"
CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"
ENV_NAME="${ENV_NAME:-reasoningqat}"

BLOCK_GPU="${BLOCK_GPU:-0}"
CONVERT_GPU="${CONVERT_GPU:-0}"
if [[ -n "${DISTILL_GPUS:-}" ]]; then
  :
elif command -v nvidia-smi >/dev/null 2>&1; then
  DISTILL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
else
  DISTILL_GPUS="0,1,2,3,4,5,6,7"
fi

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TARGET_BATCH="${TARGET_BATCH:-64}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
DATASET_SIZE="${DATASET_SIZE:-32768}"
DATASET_TYPE="${DATASET_TYPE:-openthoughts}"

usage() {
  cat <<'EOF'
Qwen3-1.7B ReasoningQAT (paper: teacher-weighted CE + forward KL + cosine)

Options:
  --wbits {2|3}          Quantization bits (default: 3)
  --stage {all|1|2|3}    Which stage to run (default: all)
  --gpus ID,ID,...       GPUs for Stage 2 (default: all visible)
  --block-gpu ID         GPU for Stage 1 block QAT (default: 0)
  --max-length N         Distill sequence length (default: 8192)
  --model PATH           Student / base model
  --teacher PATH         Teacher model for Stage 2
  --exp-name STEM        Output tag stem (default: Qwen3-1.7B → STEM-w{2|3}g128-reasoningqat)
  --train-emb            Also train embed_tokens in Stage 2
  --skip-existing        Skip a stage if its output dir already has config.json
  -h, --help             Show this help

Stage 1 matches Exp #1 / paper (sweep_0.8). Stage 2 writes *-reasoningqat.
If Exp #1 already finished Stage 1, use --stage 2 (or --skip-existing).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wbits) WBITS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --gpus) DISTILL_GPUS="$2"; shift 2 ;;
    --block-gpu) BLOCK_GPU="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --model) MODEL="$2"; MODEL_CLI=1; shift 2 ;;
    --teacher) TEACHER="$2"; TEACHER_CLI=1; shift 2 ;;
    --exp-name) EXP_NAME_STEM="$2"; shift 2 ;;
    --train-emb) TRAIN_EMB=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${WBITS}" != "2" && "${WBITS}" != "3" ]]; then
  echo "--wbits must be 2 or 3, got ${WBITS}" >&2
  exit 1
fi
if [[ "${STAGE}" != "all" && "${STAGE}" != "1" && "${STAGE}" != "2" && "${STAGE}" != "3" ]]; then
  echo "--stage must be all|1|2|3, got ${STAGE}" >&2
  exit 1
fi
if [[ "${TEACHER_CLI}" -eq 0 && "${MODEL_CLI}" -eq 1 ]]; then
  TEACHER="${MODEL}"
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 1
fi
if [[ "${MODEL}" == *4B* && "${EXP_NAME_STEM}" == "Qwen3-1.7B" ]]; then
  echo "Refusing: 4B model but --exp-name is still Qwen3-1.7B (would overwrite 1.7B outputs). Pass --exp-name Qwen3-4B." >&2
  exit 1
fi
if [[ "${MODEL}" == *4B* && "${TEACHER}" == *1.7B* ]]; then
  echo "Refusing: 4B student with 1.7B teacher ${TEACHER}. Pass --teacher to the same 4B checkpoint." >&2
  exit 1
fi

if [[ "${WBITS}" == "2" ]]; then
  WEIGHT_LR="2e-5"
  DISTILL_LR="5e-6"
  DISTILL_EPOCHS=3
else
  WEIGHT_LR="1e-5"
  DISTILL_LR="1e-6"
  DISTILL_EPOCHS=1
fi

EXP_NAME="${EXP_NAME_STEM}-w${WBITS}g${GROUP_SIZE}"
if [[ "${TRAIN_EMB}" -eq 1 ]]; then
  DISTILL_TAG="${EXP_NAME}-trainemb-reasoningqat"
else
  DISTILL_TAG="${EXP_NAME}-reasoningqat"
fi

BLOCK_DIR="${ROOT}/output/block_qat/${EXP_NAME}"
BLOCK_LOG="${ROOT}/log/block_qat/${EXP_NAME}"
DISTILL_DIR="${ROOT}/output/distill/${DISTILL_TAG}"
DISTILL_LOG="${ROOT}/log/distill/${DISTILL_TAG}"
VLLM_DIR="${ROOT}/output/vllm/${DISTILL_TAG}"

run_stage() {
  case "${STAGE}" in
    all) return 0 ;;
    "$1") return 0 ;;
    *) return 1 ;;
  esac
}

dir_ready() {
  [[ -f "$1/config.json" ]]
}

num_csv() {
  local s="$1"
  s="${s// /}"
  if [[ -z "${s}" ]]; then
    echo 0
    return
  fi
  echo "${s}" | awk -F',' '{print NF}'
}

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="$(command -v python)"
mkdir -p "${ROOT}/log" "${ROOT}/output" "${ROOT}/cache"

N_DISTILL_GPUS="$(num_csv "${DISTILL_GPUS}")"
SPLIT=$(( PER_DEVICE_BATCH * N_DISTILL_GPUS ))
if [[ "${SPLIT}" -lt 1 || $(( TARGET_BATCH % SPLIT )) -ne 0 ]]; then
  echo "effective batch ${TARGET_BATCH} must divide per_device(${PER_DEVICE_BATCH}) * n_gpus(${N_DISTILL_GPUS})" >&2
  exit 1
fi
GRAD_ACCUM=$(( TARGET_BATCH / SPLIT ))
EFFECTIVE_BATCH=$(( PER_DEVICE_BATCH * GRAD_ACCUM * N_DISTILL_GPUS ))

TRAIN_EMB_FLAG=""
if [[ "${TRAIN_EMB}" -eq 1 ]]; then
  TRAIN_EMB_FLAG="--train_emb"
fi

echo "============================================================"
echo "${EXP_NAME_STEM} ReasoningQAT (paper method)"
echo "  exp_name=${EXP_NAME}  wbits=${WBITS}  group_size=${GROUP_SIZE}  stage=${STAGE}"
echo "  model=${MODEL}"
echo "  teacher=${TEACHER}"
echo "  python=${PY}"
echo "Stage 1 GPU : ${BLOCK_GPU}   (block-wise QAT, 1 GPU)"
echo "Stage 2 GPUs: ${DISTILL_GPUS}  (${N_DISTILL_GPUS} GPU, accum=${GRAD_ACCUM}, effective_batch=${EFFECTIVE_BATCH}, max_length=${MAX_LENGTH})"
echo "  loss: 0.2 * teacher-weighted CE + 1.0 * forward_kl (top_k=20), cosine LR"
echo "  lr=${DISTILL_LR}  epochs=${DISTILL_EPOCHS}"
echo "Stage 3 GPU : ${CONVERT_GPU}"
echo "Outputs:"
echo "  block   ${BLOCK_DIR}"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

if run_stage 1; then
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && dir_ready "${BLOCK_DIR}"; then
    echo "[stage1] skip, already exists: ${BLOCK_DIR}"
  else
    echo "[stage1] block-wise QAT on GPU ${BLOCK_GPU}"
    mkdir -p "${BLOCK_DIR}" "${BLOCK_LOG}"
    CUDA_VISIBLE_DEVICES="${BLOCK_GPU}" "${PY}" main_block_qat.py \
      --model "${MODEL}" \
      --wbits "${WBITS}" \
      --group_size "${GROUP_SIZE}" \
      --calib_dataset sweep_0.8 \
      --train_size 4096 \
      --val_size 64 \
      --training_seqlen 2048 \
      --epochs 2 \
      --batch_size 2 \
      --weight_lr "${WEIGHT_LR}" \
      --quant_lr 1e-4 \
      --cache_dir "${ROOT}/cache" \
      --save_quant_dir "${BLOCK_DIR}" \
      --output_dir "${BLOCK_LOG}"
    echo "[stage1] done -> ${BLOCK_DIR}"
  fi
fi

if run_stage 2; then
  if [[ ! -f "${BLOCK_DIR}/config.json" ]]; then
    echo "[stage2] missing Stage 1 output: ${BLOCK_DIR}" >&2
    echo "         Run --stage 1 first, or reuse Exp #1 Stage 1 if present." >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && dir_ready "${DISTILL_DIR}"; then
    echo "[stage2] skip, already exists: ${DISTILL_DIR}"
  else
    echo "[stage2] ReasoningQAT e2e distill on GPUs ${DISTILL_GPUS} (effective_batch=${EFFECTIVE_BATCH})"
    mkdir -p "${DISTILL_DIR}" "${DISTILL_LOG}"
    CUDA_VISIBLE_DEVICES="${DISTILL_GPUS}" accelerate launch \
      --config_file "${ROOT}/configs/accelerate_config_multigpu.yaml" \
      --num_processes "${N_DISTILL_GPUS}" \
      --gpu_ids all \
      main_e2e_distill.py \
      --model "${BLOCK_DIR}" \
      --teacher_model "${TEACHER}" \
      --wbits "${WBITS}" \
      --group_size "${GROUP_SIZE}" \
      --epochs "${DISTILL_EPOCHS}" \
      --learning_rate "${DISTILL_LR}" \
      --lr_scheduler_type cosine \
      --gkd_beta 1.0 \
      --kl_weight 1.0 \
      --cross_entropy_weight 0.2 \
      --use_teacher_weight \
      --kd_loss_type forward_kl \
      --top_k 20 \
      --dataset_type "${DATASET_TYPE}" \
      --dataset_size "${DATASET_SIZE}" \
      --max_length "${MAX_LENGTH}" \
      --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      ${TRAIN_EMB_FLAG} \
      --save_quant_dir "${DISTILL_DIR}" \
      --output_dir "${DISTILL_LOG}"
    echo "[stage2] done -> ${DISTILL_DIR}"
  fi
fi

if run_stage 3; then
  if [[ ! -f "${DISTILL_DIR}/config.json" ]]; then
    echo "[stage3] missing Stage 2 output: ${DISTILL_DIR}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && dir_ready "${VLLM_DIR}"; then
    echo "[stage3] skip, already exists: ${VLLM_DIR}"
  else
    echo "[stage3] convert to vLLM-compatible model"
    mkdir -p "${VLLM_DIR}"
    CUDA_VISIBLE_DEVICES="${CONVERT_GPU}" "${PY}" scripts/convert_to_hf_vllm_compatible_model.py \
      --base-id "${DISTILL_DIR}" \
      --save-dir "${VLLM_DIR}" \
      --wbits "${WBITS}" \
      --group-size "${GROUP_SIZE}"
    echo "[stage3] done -> ${VLLM_DIR}"
  fi
fi

echo "============================================================"
echo "Finished stage=${STAGE}  ${DISTILL_TAG}"
echo "  block   ${BLOCK_DIR}"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
