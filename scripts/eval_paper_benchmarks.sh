#!/usr/bin/env bash
# Single eval suite (evalscope): paper tasks + GSM8K + AIME24/25.
#
#   GSM8K, AIME24, AIME25, MATH-500, LiveCodeBench,
#   MMLU-Redux, GPQA-Diamond, IFEval
#
# Paper sampling: temperature 0.6, top_k 20, max_tokens 8192
#
# Usage:
#   bash scripts/eval_paper_benchmarks.sh ./output/vllm/Qwen3-1.7B-w3g128
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_PATH="${1:-${ROOT}/output/vllm/Qwen3-1.7B-w3g128}"
WORK_DIR="${2:-${ROOT}/output/eval/Qwen3-1.7B-w3g128}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
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

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v evalscope >/dev/null 2>&1; then
  echo "installing evalscope"
  pip install -U "evalscope[ifeval]>=0.16" modelscope
fi

mkdir -p "${WORK_DIR}"
echo "eval  model=${MODEL_PATH}"
echo "  hub=modelscope  datasets: ${DATASETS[*]}"
echo "  gen: T=0.6 top_k=20 max_tokens=${MAX_TOKENS}"
echo "  work_dir=${WORK_DIR}"

EVAL_CMD=(
  evalscope eval
  --model "${MODEL_PATH}"
  --model-args '{"precision": "torch.bfloat16", "device_map": "auto"}'
  --datasets "${DATASETS[@]}"
  --dataset-hub modelscope
  --dataset-args '{"live_code_bench": {"subset_list": ["release_latest"]}}'
  --generation-config "{\"do_sample\": true, \"temperature\": 0.6, \"top_k\": 20, \"max_tokens\": ${MAX_TOKENS}}"
  --work-dir "${WORK_DIR}"
  --ignore-errors
)
if [[ -n "${LIMIT:-}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
  echo "  limit=${LIMIT}"
fi
"${EVAL_CMD[@]}"

echo "done -> ${WORK_DIR}"
