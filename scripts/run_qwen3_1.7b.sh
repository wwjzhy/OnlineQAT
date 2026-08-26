#!/usr/bin/env bash
# Qwen3-1.7B ReasoningQAT full pipeline (paper reproduction).
#
# Pipeline:
#   Stage 1  Block-wise QAT          -> 1 GPU (layer-by-layer, cannot use 8 GPUs)
#   Stage 2  End-to-end distillation -> all visible GPUs; keep effective batch 64
#   Stage 3  Convert to vLLM/HF      -> 1 GPU (or CPU)
#
# 8x A100-80GB (paper-faithful, faster wall clock):
#   Stage 2: 8 GPU, per_device_batch=1, grad_accum=8, max_length=8192
#
# 8x A100-40GB:
#   Stage 1 is fine. Stage 2 is tight at 8192 (student+teacher+logits ~30-38GB).
#   Keep per_device_batch=1. If OOM, drop to --max-length 4096.
#   More GPUs do not reduce per-GPU memory (DDP replicates both models).
#
# Usage:
#   bash scripts/run_qwen3_1.7b.sh --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_qwen3_1.7b.sh --gpus 0,1,2,3,4,5,6,7 --max-length 4096
#   bash scripts/run_qwen3_1.7b.sh --wbits 3
#   bash scripts/run_qwen3_1.7b.sh --stage 1
#   bash scripts/run_qwen3_1.7b.sh --stage 2 --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_qwen3_1.7b.sh --train-emb
#   bash scripts/run_qwen3_1.7b.sh --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ---------------------------------------------------------------------------
# Defaults (paper: Qwen3-1.7B W2)
# ---------------------------------------------------------------------------
WBITS=2
GROUP_SIZE=128
STAGE="all"
TRAIN_EMB=0
SKIP_EXISTING=0

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

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Qwen3-1.7B ReasoningQAT pipeline

Options:
  --wbits {2|3}          Quantization bits (default: 2)
  --stage {all|1|2|3}    Which stage to run (default: all)
  --gpus ID,ID,...       GPUs for Stage 2 (default: all visible, e.g. 0-7)
  --block-gpu ID         GPU for Stage 1 block QAT (default: 0)
  --max-length N         Distill sequence length (default: 8192; use 4096 on A100-40GB if OOM)
  --model PATH           Student / base model (default: /zju_0038/zq/models/Qwen3-1.7B)
  --teacher PATH         Teacher model for Stage 2 (default: same as --model)
  --train-emb            Also train embed_tokens in Stage 2
  --skip-existing        Skip a stage if its output dir already has config.json
  -h, --help             Show this help

Environment overrides:
  MODEL_PATH, TEACHER_MODEL, BLOCK_GPU, DISTILL_GPUS, CONVERT_GPU
  CONDA_ROOT, ENV_NAME, PER_DEVICE_BATCH, TARGET_BATCH, MAX_LENGTH
  DATASET_SIZE, HF_ENDPOINT, HF_HOME

GPU notes:
  Stage 1 is sequential and always uses 1 GPU. 40GB is enough for 1.7B.
  Stage 2 keeps effective batch 64:
    gradient_accumulation_steps = 64 / (per_device_batch * N)
  8x A100: N=8, accum=8  (paper used N=4, accum=16)
  A100-40GB: keep per_device=1; if Stage 2 OOM, --max-length 4096
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wbits) WBITS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --gpus) DISTILL_GPUS="$2"; shift 2 ;;
    --block-gpu) BLOCK_GPU="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
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
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 1
fi

# Paper hyperparams for Qwen3-1.7B
if [[ "${WBITS}" == "2" ]]; then
  WEIGHT_LR="2e-5"
  DISTILL_LR="5e-6"
  DISTILL_EPOCHS=3
else
  WEIGHT_LR="1e-5"
  DISTILL_LR="1e-6"
  DISTILL_EPOCHS=1
fi

EXP_NAME="Qwen3-1.7B-w${WBITS}g${GROUP_SIZE}"
if [[ "${TRAIN_EMB}" -eq 1 ]]; then
  DISTILL_TAG="${EXP_NAME}-trainemb"
else
  DISTILL_TAG="${EXP_NAME}"
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

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
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
echo "Qwen3-1.7B ReasoningQAT"
echo "  wbits=${WBITS}  group_size=${GROUP_SIZE}  stage=${STAGE}"
echo "  model=${MODEL}"
echo "  teacher=${TEACHER}"
echo "  python=${PY}"
echo "Stage 1 GPU : ${BLOCK_GPU}   (block-wise QAT, 1 GPU)"
echo "Stage 2 GPUs: ${DISTILL_GPUS}  (${N_DISTILL_GPUS} GPU, accum=${GRAD_ACCUM}, effective_batch=${EFFECTIVE_BATCH}, max_length=${MAX_LENGTH})"
echo "  paper: 4 GPU * accum 16 = 64; this run: ${N_DISTILL_GPUS} GPU * accum ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}"
echo "Stage 3 GPU : ${CONVERT_GPU}"
echo "Outputs:"
echo "  block   ${BLOCK_DIR}"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

# ---------------------------------------------------------------------------
# Stage 1: Block-wise QAT (1 GPU)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stage 2: End-to-end distillation (paper: 4 GPU)
# ---------------------------------------------------------------------------
if run_stage 2; then
  if [[ ! -f "${BLOCK_DIR}/config.json" ]]; then
    echo "[stage2] missing Stage 1 output: ${BLOCK_DIR}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && dir_ready "${DISTILL_DIR}"; then
    echo "[stage2] skip, already exists: ${DISTILL_DIR}"
  else
    echo "[stage2] e2e distill on GPUs ${DISTILL_GPUS} (effective_batch=${EFFECTIVE_BATCH})"
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
      --kl_weight 1.0 \
      --cross_entropy_weight 0.2 \
      --kd_loss_type jsd \
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

# ---------------------------------------------------------------------------
# Stage 3: Convert to vLLM-compatible HF
# ---------------------------------------------------------------------------
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
echo "Finished stage=${STAGE}  ${EXP_NAME}"
echo "  block   ${BLOCK_DIR}"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
