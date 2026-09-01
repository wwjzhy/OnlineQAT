#!/usr/bin/env bash
# Qwen3-1.7B W2 PV-OPD FullPair.
#
# Reuses Exp #4 Stage 1, rolls out W2, verifies token signals with a shared-
# clipping W4 probe, and updates full master weights plus quantizer scales.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WBITS=2
PROBE_BITS="${PV_PROBE_BITS:-4}"
GROUP_SIZE=128
STAGE=2
DISTILL_GPUS="${DISTILL_GPUS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-5}"
GATE_MODE="${PV_GATE_MODE:-full}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
DATASET_SIZE="${DATASET_SIZE:-32768}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TARGET_BATCH="${TARGET_BATCH:-64}"

MODEL="${MODEL_PATH:-/zju_0038/zq/models/Qwen3-1.7B}"
TEACHER="${TEACHER_MODEL:-${MODEL}}"
CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"
ENV_NAME="${ENV_NAME:-reasoningqat}"
CONVERT_GPU="${CONVERT_GPU:-0}"

usage() {
  cat <<'EOF'
Qwen3-1.7B W2 PV-OPD FullPair (W4 precision probe).

Requires Exp #4 Stage 1:
  output/block_qat/Qwen3-1.7B-w2g128

Options:
  --stage {2|3}          2=train PV-OPD, 3=convert final model (default: 2)
  --gpus ID,ID,...       Stage 2 GPUs (default: 0,1,2,3,4,5,6,7)
  --probe-bits N         Probe precision (default: 4)
  --max-steps N          Optimizer-step cap (default: 100)
  --save-steps N         Snapshot interval (default: 5)
  --gate-mode MODE       full|sign|shuffled (default: full)
  --max-length N         Total prompt+rollout cap (default: 8192)
  --model PATH           BF16 base model path
  --teacher PATH         Frozen BF16 teacher path
  -h, --help

The training path updates full target master weights and quantizer scales;
zero-points and embeddings remain frozen. Evaluation is run after 8-GPU
training with scripts/eval_distill_checkpoints.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --gpus) DISTILL_GPUS="$2"; shift 2 ;;
    --probe-bits) PROBE_BITS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --gate-mode) GATE_MODE="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${STAGE}" != "2" && "${STAGE}" != "3" ]]; then
  echo "--stage must be 2 or 3, got ${STAGE}" >&2
  exit 1
fi
if [[ "${PROBE_BITS}" -le "${WBITS}" ]]; then
  echo "--probe-bits must be greater than W${WBITS}" >&2
  exit 1
fi
if [[ "${GATE_MODE}" != "full" && "${GATE_MODE}" != "sign" && "${GATE_MODE}" != "shuffled" ]]; then
  echo "--gate-mode must be full|sign|shuffled" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 1
fi

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
N_GPUS="$(echo "${DISTILL_GPUS// /}" | awk -F',' '{print NF}')"
SPLIT=$(( PER_DEVICE_BATCH * N_GPUS ))
if [[ "${SPLIT}" -lt 1 || $(( TARGET_BATCH % SPLIT )) -ne 0 ]]; then
  echo "effective batch ${TARGET_BATCH} must divide per_device(${PER_DEVICE_BATCH}) * n_gpus(${N_GPUS})" >&2
  exit 1
fi
GRAD_ACCUM=$(( TARGET_BATCH / SPLIT ))

EXP_NAME="Qwen3-1.7B-w${WBITS}g${GROUP_SIZE}"
TAG="${EXP_NAME}-pv-opd"
if [[ "${GATE_MODE}" != "full" ]]; then
  TAG="${TAG}-${GATE_MODE}"
fi
BLOCK_DIR="${ROOT}/output/block_qat/${EXP_NAME}"
DISTILL_DIR="${ROOT}/output/distill/${TAG}"
DISTILL_LOG="${ROOT}/log/distill/${TAG}"
VLLM_DIR="${ROOT}/output/vllm/${TAG}"

echo "============================================================"
echo "PV-OPD FullPair: W${WBITS} target / W${PROBE_BITS} probe"
echo "stage=${STAGE} gate=${GATE_MODE} GPUs=${DISTILL_GPUS}"
echo "max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} max_length=${MAX_LENGTH}"
echo "effective_batch=${TARGET_BATCH} grad_accum=${GRAD_ACCUM}"
echo "block=${BLOCK_DIR}"
echo "distill=${DISTILL_DIR}"
echo "============================================================"

if [[ "${STAGE}" == "2" ]]; then
  if [[ ! -f "${BLOCK_DIR}/config.json" ]]; then
    echo "Missing Exp #4 Stage 1 checkpoint: ${BLOCK_DIR}" >&2
    exit 1
  fi
  mkdir -p "${DISTILL_DIR}" "${DISTILL_LOG}"
  rm -f "${DISTILL_DIR}/.train_done"
  CUDA_VISIBLE_DEVICES="${DISTILL_GPUS}" accelerate launch \
    --config_file "${ROOT}/configs/accelerate_config_multigpu.yaml" \
    --num_processes "${N_GPUS}" \
    --gpu_ids all \
    main_e2e_distill.py \
    --model "${BLOCK_DIR}" \
    --teacher_model "${TEACHER}" \
    --wbits "${WBITS}" \
    --group_size "${GROUP_SIZE}" \
    --epochs 3 \
    --max_steps "${MAX_STEPS}" \
    --learning_rate 5e-6 \
    --kl_weight 1.0 \
    --cross_entropy_weight 0.0 \
    --pv_opd \
    --pv_probe_bits "${PROBE_BITS}" \
    --pv_gate_mode "${GATE_MODE}" \
    --pv_gate_max 2.0 \
    --pv_adv_clip_warmup_steps 10 \
    --dataset_type openthoughts \
    --dataset_size "${DATASET_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_steps "${SAVE_STEPS}" \
    --save_quant_dir "${DISTILL_DIR}" \
    --output_dir "${DISTILL_LOG}"
  touch "${DISTILL_DIR}/.ready" "${DISTILL_DIR}/.train_done"
  echo "PV-OPD Stage 2 done -> ${DISTILL_DIR}"
fi

if [[ "${STAGE}" == "3" ]]; then
  if [[ ! -f "${DISTILL_DIR}/config.json" ]]; then
    echo "Missing PV-OPD Stage 2 output: ${DISTILL_DIR}" >&2
    exit 1
  fi
  mkdir -p "${VLLM_DIR}"
  CUDA_VISIBLE_DEVICES="${CONVERT_GPU}" "${PY}" scripts/convert_to_hf_vllm_compatible_model.py \
    --base-id "${DISTILL_DIR}" \
    --save-dir "${VLLM_DIR}" \
    --wbits "${WBITS}" \
    --group-size "${GROUP_SIZE}"
  echo "PV-OPD Stage 3 done -> ${VLLM_DIR}"
fi
