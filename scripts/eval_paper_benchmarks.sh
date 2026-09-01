#!/usr/bin/env bash
# Single eval suite (evalscope + vLLM): paper tasks + GSM8K + AIME24/25.
#
#   GSM8K, AIME24, AIME25, MATH-500, LiveCodeBench,
#   MMLU-Redux, GPQA-Diamond, IFEval
#
# Paper sampling: temperature 0.6, top_k 20, max_tokens 8192
#
# Starts `vllm serve` on localhost, then evalscope --eval-type openai_api.
# The server is stopped on exit.
#
# Usage:
#   bash scripts/eval_paper_benchmarks.sh ./output/vllm/Qwen3-1.7B-w3g128
#
# Optional env:
#   MAX_TOKENS          default 8192
#   LIMIT               smoke: LIMIT=1
#   VLLM_PORT           default 8000
#   TP                  tensor parallel size, default 1
#   GPU_UTIL            vLLM gpu-memory-utilization, default 0.9
#   MAX_MODEL_LEN       default 32768
#   EVAL_BATCH_SIZE     concurrent API requests, default 32
#   EVAL_DATASETS       space-separated dataset override
#   VLLM_API_URL        skip serve; eval against this OpenAI base (…/v1)
#   SERVED_MODEL_NAME   must match vLLM --served-model-name if VLLM_API_URL is set
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_PATH="${1:-${ROOT}/output/vllm/Qwen3-1.7B-w3g128}"
WORK_DIR="${2:-${ROOT}/output/eval/Qwen3-1.7B-w3g128}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
# Eval datasets come from ModelScope (no HF gated token for GPQA-Diamond).
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/hf_cache/modelscope}"

DATASETS=(
  gsm8k
  aime24
  aime25
  math_500
  live_code_bench
  mmlu_redux
  gpqa_diamond
  ifeval
)
if [[ -n "${EVAL_DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS=(${EVAL_DATASETS})
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v evalscope >/dev/null 2>&1; then
  echo "installing evalscope"
  pip install -U "evalscope[ifeval]>=0.16" modelscope
fi

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "installing vllm"
  pip install -U vllm
fi

mkdir -p "${WORK_DIR}"

VLLM_PID=""
stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "stopping vllm pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap stop_vllm EXIT INT TERM

wait_for_vllm() {
  local base_url="$1"
  local pid="$2"
  local i
  for i in $(seq 1 120); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vllm serve died; see ${WORK_DIR}/vllm_serve.log" >&2
      return 1
    fi
    if python - "${base_url}" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/models"
try:
    urllib.request.urlopen(url, timeout=2)
except Exception:
    sys.exit(1)
PY
    then
      echo "vllm ready  ${base_url}"
      return 0
    fi
    sleep 5
  done
  echo "vllm not ready after timeout; see ${WORK_DIR}/vllm_serve.log" >&2
  return 1
}

if [[ -n "${VLLM_API_URL:-}" ]]; then
  API_URL="${VLLM_API_URL}"
  echo "reuse vllm  ${API_URL}  model=${SERVED_MODEL_NAME}"
else
  API_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  echo "vllm serve  model=${MODEL_PATH}"
  echo "  served_name=${SERVED_MODEL_NAME}  port=${VLLM_PORT}  tp=${TP}"
  echo "  gpu_util=${GPU_UTIL}  max_model_len=${MAX_MODEL_LEN}"
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${VLLM_PORT}" \
    --disable-log-requests \
    >"${WORK_DIR}/vllm_serve.log" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm "${API_URL}" "${VLLM_PID}"
fi

echo "eval  model=${SERVED_MODEL_NAME}"
echo "  hub=modelscope  datasets: ${DATASETS[*]}"
echo "  gen: T=0.6 top_k=20 max_tokens=${MAX_TOKENS}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}  work_dir=${WORK_DIR}"

EVAL_CMD=(
  evalscope eval
  --model "${SERVED_MODEL_NAME}"
  --eval-type openai_api
  --api-url "${API_URL}"
  --api-key EMPTY
  --datasets "${DATASETS[@]}"
  --dataset-hub modelscope
  --dataset-args '{"live_code_bench": {"subset_list": ["release_latest"]}}'
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --generation-config "{\"temperature\": 0.6, \"top_k\": 20, \"max_tokens\": ${MAX_TOKENS}, \"timeout\": 3600, \"extra_body\": {\"top_k\": 20}}"
  --work-dir "${WORK_DIR}"
  --ignore-errors
)
if [[ -n "${LIMIT:-}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
  echo "  limit=${LIMIT}"
fi
"${EVAL_CMD[@]}"

echo "done -> ${WORK_DIR}"
