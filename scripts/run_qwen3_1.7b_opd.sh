#!/usr/bin/env bash
# Qwen3-1.7B ReasoningQAT Stage 2/3 with OPD (student rollout + sampled reverse KL).
#
# Same data and hparams as scripts/run_qwen3_1.7b.sh. Difference:
#   GKD  offline JSD+CE on OpenThoughts gold completions
#   OPD  student generates, then sampled reverse-KL policy gradient (no CE / gold SFT)
#
# Stage 1 is NOT run here. Train it once with:
#   bash scripts/run_qwen3_1.7b.sh --wbits 3 --stage 1
# then:
#   bash scripts/run_qwen3_1.7b_opd.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7
#
# Outputs go to *-opd dirs so they do not overwrite the GKD run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ---------------------------------------------------------------------------
# Defaults (match GKD script)
# ---------------------------------------------------------------------------
WBITS=2
GROUP_SIZE=128
STAGE="all"
TRAIN_EMB=0
SKIP_EXISTING=0
SAVE_STEPS="${SAVE_STEPS:-0}"
EVAL_GPU="${EVAL_GPU:-}"
MAX_STEPS="${MAX_STEPS:-0}"
# Empty = use the bitwidth default below. Exp #7 overrides with --lr 1e-6.
DISTILL_LR_OVERRIDE=""
WARMUP_STEPS=""
WARMUP_START_LR=""

MODEL="${MODEL_PATH:-/zju_0038/zq/models/Qwen3-1.7B}"
TEACHER="${TEACHER_MODEL:-${MODEL}}"
CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"
ENV_NAME="${ENV_NAME:-reasoningqat}"

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
Qwen3-1.7B ReasoningQAT pipeline (OPD / on-policy JSD)

Requires Stage 1 from scripts/run_qwen3_1.7b.sh. This script runs Stage 2+3 only.

Options:
  --wbits {2|3}          Quantization bits (default: 2)
  --stage {all|2|3}      2=distill, 3=convert, all=2 then 3 (default: all)
  --gpus ID,ID,...       GPUs for Stage 2 (default: all visible)
  --max-length N         Distill sequence length (default: 8192)
  --model PATH           Base model (only used to check the path exists)
  --teacher PATH         Teacher model for Stage 2 (default: same as --model)
  --train-emb            Also train embed_tokens in Stage 2
  --skip-existing        Skip a stage if its output dir already has config.json
  --save-steps N         Save a convert-ready snapshot every N Stage 2 steps (0=off)
  --max-steps N          Cap Stage 2 optimizer steps (0=use epochs; Exp #5 uses 100)
  --lr RATE              Stage 2 learning rate (default: 5e-6 for W2, 1e-6 for W3)
  --warmup-steps N       Fixed warmup steps (default: warmup_ratio=0.2)
  --warmup-start-lr RATE LR at step 0 during warmup (default: 0)
  --eval-gpu ID          GPU for convert+evalscope watcher (requires --save-steps > 0)
  -h, --help             Show this help

Environment overrides: same as scripts/run_qwen3_1.7b.sh
  DISTILL_LR             Same as --lr if the flag is omitted
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wbits) WBITS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --gpus) DISTILL_GPUS="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    --train-emb) TRAIN_EMB=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --lr) DISTILL_LR_OVERRIDE="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --warmup-start-lr) WARMUP_START_LR="$2"; shift 2 ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "${WBITS}" != "2" && "${WBITS}" != "3" ]]; then
  echo "--wbits must be 2 or 3, got ${WBITS}" >&2
  exit 1
fi
if [[ "${STAGE}" != "all" && "${STAGE}" != "2" && "${STAGE}" != "3" ]]; then
  echo "OPD script --stage must be all|2|3 (Stage 1 is scripts/run_qwen3_1.7b.sh), got ${STAGE}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 1
fi

if [[ "${WBITS}" == "2" ]]; then
  DEFAULT_DISTILL_LR="5e-6"
  DISTILL_EPOCHS=3
else
  DEFAULT_DISTILL_LR="1e-6"
  DISTILL_EPOCHS=1
fi
# Flag wins over env; env wins over bitwidth default.
if [[ -n "${DISTILL_LR_OVERRIDE}" ]]; then
  DISTILL_LR="${DISTILL_LR_OVERRIDE}"
elif [[ -n "${DISTILL_LR:-}" ]]; then
  :
else
  DISTILL_LR="${DEFAULT_DISTILL_LR}"
fi

EXP_NAME="Qwen3-1.7B-w${WBITS}g${GROUP_SIZE}"
DISTILL_TAG="${EXP_NAME}-opd"
if [[ "${TRAIN_EMB}" -eq 1 ]]; then
  DISTILL_TAG="${EXP_NAME}-trainemb-opd"
fi
# Non-default LR gets its own output dir so Exp #5 (5e-6) is not overwritten.
if [[ "${DISTILL_LR}" != "${DEFAULT_DISTILL_LR}" ]]; then
  LR_TAG="$(printf '%s' "${DISTILL_LR}" | tr '[:upper:]' '[:lower:]' | sed 's/[.]/_/g')"
  DISTILL_TAG="${DISTILL_TAG}-lr${LR_TAG}"
fi
if [[ -n "${WARMUP_STEPS}" ]]; then
  DISTILL_TAG="${DISTILL_TAG}-wu${WARMUP_STEPS}"
fi
if [[ -n "${WARMUP_START_LR}" ]]; then
  WS_TAG="$(printf '%s' "${WARMUP_START_LR}" | tr '[:upper:]' '[:lower:]' | sed 's/[.]/_/g')"
  DISTILL_TAG="${DISTILL_TAG}-ws${WS_TAG}"
fi

BLOCK_DIR="${ROOT}/output/block_qat/${EXP_NAME}"
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

SAVE_STEPS_FLAG=""
if [[ "${SAVE_STEPS}" -gt 0 ]]; then
  SAVE_STEPS_FLAG="--save_steps ${SAVE_STEPS}"
fi

MAX_STEPS_FLAG=""
if [[ "${MAX_STEPS}" -gt 0 ]]; then
  MAX_STEPS_FLAG="--max_steps ${MAX_STEPS}"
fi

WARMUP_FLAGS=()
if [[ -n "${WARMUP_STEPS}" ]]; then
  WARMUP_FLAGS+=(--warmup_steps "${WARMUP_STEPS}")
fi
if [[ -n "${WARMUP_START_LR}" ]]; then
  WARMUP_FLAGS+=(--warmup_start_lr "${WARMUP_START_LR}")
fi

if [[ -n "${EVAL_GPU}" ]]; then
  case ",${DISTILL_GPUS}," in
    *",${EVAL_GPU},"*)
      echo "--eval-gpu ${EVAL_GPU} overlaps Stage 2 GPUs ${DISTILL_GPUS}" >&2
      exit 1
      ;;
  esac
  if [[ "${SAVE_STEPS}" -le 0 ]]; then
    echo "--eval-gpu requires --save-steps N with N>0" >&2
    exit 1
  fi
fi

echo "============================================================"
echo "Qwen3-1.7B ReasoningQAT (OPD, sampled reverse KL on student rollouts)"
echo "  wbits=${WBITS}  group_size=${GROUP_SIZE}  stage=${STAGE}"
echo "  model=${MODEL}"
echo "  teacher=${TEACHER}"
echo "  python=${PY}"
echo "Stage 2 GPUs: ${DISTILL_GPUS}  (${N_DISTILL_GPUS} GPU, accum=${GRAD_ACCUM}, effective_batch=${EFFECTIVE_BATCH}, max_length=${MAX_LENGTH})"
echo "  student generate then sampled reverse-KL policy gradient; no CE"
echo "  lr=${DISTILL_LR} (default ${DEFAULT_DISTILL_LR})  warmup_steps=${WARMUP_STEPS:-ratio0.2}  warmup_start_lr=${WARMUP_START_LR:-0}"
echo "  save_steps=${SAVE_STEPS}  max_steps=${MAX_STEPS:-epochs}  eval_gpu=${EVAL_GPU:-none}"
echo "Stage 3 GPU : ${CONVERT_GPU}"
echo "Inputs / outputs:"
echo "  block   ${BLOCK_DIR}   (from run_qwen3_1.7b.sh Stage 1)"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

# ---------------------------------------------------------------------------
# Stage 2: On-policy sampled reverse KL (slime-compatible objective)
# ---------------------------------------------------------------------------
if run_stage 2; then
  if [[ ! -f "${BLOCK_DIR}/config.json" ]]; then
    echo "[stage2] missing Stage 1 output: ${BLOCK_DIR}" >&2
    echo "         Run: bash scripts/run_qwen3_1.7b.sh --wbits ${WBITS} --stage 1" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && dir_ready "${DISTILL_DIR}"; then
    echo "[stage2] skip, already exists: ${DISTILL_DIR}"
  else
    echo "[stage2] OPD e2e distill on GPUs ${DISTILL_GPUS} (effective_batch=${EFFECTIVE_BATCH})"
    mkdir -p "${DISTILL_DIR}" "${DISTILL_LOG}"
    rm -f "${DISTILL_DIR}/.train_done"
    EVAL_WATCHER_PID=""
    mark_train_done() {
      mkdir -p "${DISTILL_DIR}"
      touch "${DISTILL_DIR}/.train_done"
    }
    trap mark_train_done EXIT
    if [[ -n "${EVAL_GPU}" ]]; then
      echo "[stage2] eval watcher on GPU ${EVAL_GPU} (every ${SAVE_STEPS} steps)"
      mkdir -p "${ROOT}/log/eval/${DISTILL_TAG}"
      bash "${ROOT}/scripts/eval_distill_checkpoints.sh" \
        --watch-dir "${DISTILL_DIR}" \
        --wbits "${WBITS}" \
        --group-size "${GROUP_SIZE}" \
        --eval-gpu "${EVAL_GPU}" \
        --tag "${DISTILL_TAG}" \
        >>"${ROOT}/log/eval/${DISTILL_TAG}/watcher.log" 2>&1 &
      EVAL_WATCHER_PID="$!"
      echo "[stage2] eval watcher pid=${EVAL_WATCHER_PID}"
    fi
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
      --cross_entropy_weight 0.0 \
      --opd \
      --dataset_type "${DATASET_TYPE}" \
      --dataset_size "${DATASET_SIZE}" \
      --max_length "${MAX_LENGTH}" \
      --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      ${TRAIN_EMB_FLAG} \
      ${SAVE_STEPS_FLAG} \
      ${MAX_STEPS_FLAG} \
      ${WARMUP_FLAGS[@]+"${WARMUP_FLAGS[@]}"} \
      --save_quant_dir "${DISTILL_DIR}" \
      --output_dir "${DISTILL_LOG}"
    touch "${DISTILL_DIR}/.ready" "${DISTILL_DIR}/.train_done"
    echo "[stage2] done -> ${DISTILL_DIR}"
    if [[ -n "${EVAL_WATCHER_PID}" ]]; then
      echo "[stage2] eval watcher still running pid=${EVAL_WATCHER_PID} (log/eval/${DISTILL_TAG}/watcher.log)"
    fi
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
echo "Finished OPD stage=${STAGE}  ${DISTILL_TAG}"
echo "  block   ${BLOCK_DIR}"
echo "  distill ${DISTILL_DIR}"
echo "  vllm    ${VLLM_DIR}"
echo "============================================================"
