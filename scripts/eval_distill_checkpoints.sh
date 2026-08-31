#!/usr/bin/env bash
# Convert + evalscope each quantized snapshot under a distill dir.
#
# Watches:
#   $WATCH_DIR/checkpoint-<step>/   (written every --save-steps during Stage 2)
#   $WATCH_DIR/                     (final save after training, when .train_done exists)
#
# After a successful eval, delete the vLLM convert dir (and the distill snapshot
# unless KEEP_CHECKPOINTS=1) so 5-step dumps do not fill the disk. Eval reports stay.
#
# Usage:
#   bash scripts/eval_distill_checkpoints.sh \
#     --watch-dir ./output/distill/Qwen3-1.7B-w2g128-opd \
#     --wbits 2 --eval-gpu 2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WATCH_DIR=""
WBITS=2
GROUP_SIZE=128
EVAL_GPU="${EVAL_GPU:-0}"
POLL_SECS="${POLL_SECS:-30}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
TAG=""

usage() {
  cat <<'EOF'
Watch distill checkpoints, convert to vLLM, run eval_paper_benchmarks.sh.

Options:
  --watch-dir PATH     Distill output dir (required)
  --wbits {2|3}        Quantization bits (default: 2)
  --group-size N       Group size (default: 128)
  --eval-gpu ID        GPU for convert + vLLM (default: 0)
  --tag NAME           Eval/vLLM subdir name (default: basename of --watch-dir)
  --poll-secs N        Sleep between scans (default: 30)
  -h, --help

Environment:
  KEEP_CHECKPOINTS=1   Do not delete distill snapshots after eval
  LIMIT, MAX_TOKENS, VLLM_PORT, ...  forwarded to eval_paper_benchmarks.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir) WATCH_DIR="$2"; shift 2 ;;
    --wbits) WBITS="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --poll-secs) POLL_SECS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${WATCH_DIR}" ]]; then
  echo "--watch-dir is required" >&2
  exit 1
fi
WATCH_DIR="$(cd "${WATCH_DIR}" 2>/dev/null && pwd || echo "${WATCH_DIR}")"
if [[ -z "${TAG}" ]]; then
  TAG="$(basename "${WATCH_DIR}")"
fi

CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"
ENV_NAME="${ENV_NAME:-reasoningqat}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export VLLM_PORT="${VLLM_PORT:-$((8100 + EVAL_GPU))}"

PY="$(command -v python)"
mkdir -p "${ROOT}/output/vllm/${TAG}" "${ROOT}/output/eval/${TAG}" "${ROOT}/log/eval/${TAG}"
LOG="${ROOT}/log/eval/${TAG}/watcher.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

ckpt_ready() {
  [[ -f "$1/config.json" && -f "$1/.ready" ]]
}

eval_one() {
  local src="$1"
  local name="$2"
  local vllm_dir="${ROOT}/output/vllm/${TAG}/${name}"
  local eval_dir="${ROOT}/output/eval/${TAG}/${name}"
  local marker="$1/.eval_done"
  local lock="$1/.eval_lock"

  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if [[ -f "${lock}" ]]; then
    local lock_pid
    lock_pid="$(cat "${lock}" 2>/dev/null || true)"
    if [[ -n "${lock_pid}" ]] && kill -0 "${lock_pid}" 2>/dev/null; then
      return 0
    fi
    rm -f "${lock}"
  fi
  echo "$$" > "${lock}"

  set +e
  log "convert ${name} from ${src}"
  mkdir -p "${vllm_dir}" "${eval_dir}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" scripts/convert_to_hf_vllm_compatible_model.py \
    --base-id "${src}" \
    --save-dir "${vllm_dir}" \
    --wbits "${WBITS}" \
    --group-size "${GROUP_SIZE}"
  convert_rc=$?
  if [[ "${convert_rc}" -ne 0 ]]; then
    log "FAIL convert ${name} rc=${convert_rc}"
    rm -f "${lock}"
    set -e
    return 0
  fi

  log "eval ${name} -> ${eval_dir}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" bash scripts/eval_paper_benchmarks.sh \
    "${vllm_dir}" \
    "${eval_dir}"
  eval_rc=$?
  set -e
  if [[ "${eval_rc}" -ne 0 ]]; then
    log "FAIL eval ${name} rc=${eval_rc}"
    rm -f "${lock}"
    return 0
  fi

  date '+%F %T' > "${marker}"
  rm -f "${lock}"
  log "done ${name}"

  rm -rf "${vllm_dir}"
  if [[ "${KEEP_CHECKPOINTS}" != "1" && "${name}" != "final" ]]; then
    rm -rf "${src}"
    log "deleted distill snapshot ${name} (KEEP_CHECKPOINTS=0); eval stays at ${eval_dir}"
  fi
}

log "watch ${WATCH_DIR}  wbits=${WBITS}  eval_gpu=${EVAL_GPU}  vllm_port=${VLLM_PORT}"

pending() {
  local any=0
  local ckpt
  shopt -s nullglob
  for ckpt in "${WATCH_DIR}"/checkpoint-*; do
    if [[ -d "${ckpt}" ]] && ckpt_ready "${ckpt}" && [[ ! -f "${ckpt}/.eval_done" ]]; then
      any=1
      break
    fi
  done
  shopt -u nullglob
  if [[ -f "${WATCH_DIR}/.train_done" && -f "${WATCH_DIR}/config.json" && ! -f "${WATCH_DIR}/.eval_done" ]]; then
    any=1
  fi
  echo "${any}"
}

while true; do
  shopt -s nullglob
  for ckpt in "${WATCH_DIR}"/checkpoint-*; do
    if [[ -d "${ckpt}" ]] && ckpt_ready "${ckpt}"; then
      eval_one "${ckpt}" "$(basename "${ckpt}")"
    fi
  done
  shopt -u nullglob

  if [[ -f "${WATCH_DIR}/.train_done" && -f "${WATCH_DIR}/config.json" ]]; then
    if [[ ! -f "${WATCH_DIR}/.ready" ]]; then
      touch "${WATCH_DIR}/.ready"
    fi
    eval_one "${WATCH_DIR}" "final"
  fi

  if [[ -f "${WATCH_DIR}/.train_done" && "$(pending)" == "0" ]]; then
    log "train done and no pending checkpoints; watcher exit"
    exit 0
  fi
  sleep "${POLL_SECS}"
done
