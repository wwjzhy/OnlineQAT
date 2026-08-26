#!/usr/bin/env bash
# Create a CUDA 12.9 conda env that can run OnlineQAT / ReasoningQAT scripts.
#
# Why not `pip install -r requirements.txt`:
#   - that file pins torch<2.5 and triton<2.4, which have no cu129 wheels
#   - flash-attn must be built against the torch just installed
#   - this machine's pip.conf points at dead pypi.ngc.nvidia.com
#
# Usage:
#   bash scripts/setup_env_cu129.sh
#   conda activate reasoningqat
#
# Override:
#   CONDA_ROOT=... ENV_NAME=... bash scripts/setup_env_cu129.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"
ENV_NAME="${ENV_NAME:-reasoningqat}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
MAX_JOBS="${MAX_JOBS:-8}"
# A800=8.0, H800=9.0. Cluster CUDA 12.9 nodes typically cover both.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Ignore user/global pip.conf (NGC extra-index is broken here).
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_EXTRA_INDEX_URL || true

CONDA="${CONDA_ROOT}/bin/conda"
if [[ ! -x "${CONDA}" ]]; then
  echo "conda not found at ${CONDA}" >&2
  exit 1
fi

echo "==> conda env ${ENV_NAME} (python ${PYTHON_VERSION})"
if "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "    exists, reuse"
else
  "${CONDA}" create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" \
    -c conda-forge --override-channels
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
PY="$(command -v python)"
PIP="$(command -v pip)"
echo "    python=${PY}"
"${PY}" -V

echo "==> pip tooling"
"${PIP}" install -U pip setuptools wheel

echo "==> torch ${TORCH_VERSION} from ${TORCH_INDEX}"
"${PIP}" install \
  "torch==${TORCH_VERSION}" "torchvision==0.23.0" \
  --index-url "${TORCH_INDEX}"

"${PY}" - <<'PY'
import torch
print(f"    torch={torch.__version__}  cuda={torch.version.cuda}  cuda_built={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("    note: GPU not visible here (driver/toolkit mismatch is OK if you pack this env for a CUDA 12.9 node)")
else:
    print(f"    device={torch.cuda.get_device_name(0)}")
PY

echo "==> OnlineQAT python deps (no torch/flash-attn/triton pins)"
"${PIP}" install -r "${ROOT}/requirements-cu129.txt"

echo "==> CUDA nvcc for flash-attn (pip nvidia-* if system nvcc missing)"
if ! command -v nvcc >/dev/null 2>&1; then
  "${PIP}" install "nvidia-cuda-nvcc-cu12>=12.8" "nvidia-cuda-runtime-cu12>=12.8"
  NVCC_DIR="$("${PY}" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("nvidia")
print(Path(spec.submodule_search_locations[0]).resolve())
PY
)"
  # nvidia-cuda-nvcc-cu12 layout: site-packages/nvidia/cuda_nvcc/bin/nvcc
  if [[ -x "${NVCC_DIR}/cuda_nvcc/bin/nvcc" ]]; then
    export CUDA_HOME="${NVCC_DIR}/cuda_nvcc"
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
  # runtime headers sometimes live under nvidia/cuda_runtime
  if [[ -d "${NVCC_DIR}/cuda_runtime" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${NVCC_DIR}/cuda_nvcc}"
    export CPATH="${NVCC_DIR}/cuda_runtime/include:${CPATH:-}"
  fi
fi
echo "    nvcc=$(command -v nvcc || echo missing)"
nvcc --version 2>/dev/null | tail -n 1 || true

echo "==> flash-attn ${FLASH_ATTN_VERSION} (build against current torch)"
# Prefer a matching wheel; fall back to source build.
set +e
"${PIP}" install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
fa_status=$?
if [[ "${fa_status}" -ne 0 ]]; then
  echo "    ${FLASH_ATTN_VERSION} failed, trying 2.8.3.post1 (still FA2, works with attn_implementation=flash_attention_2)"
  "${PIP}" install "flash-attn==2.8.3.post1" --no-build-isolation --no-cache-dir
  fa_status=$?
fi
set -e
if [[ "${fa_status}" -ne 0 ]]; then
  echo "flash-attn install failed" >&2
  exit 1
fi

echo "==> smoke import"
"${PY}" - <<'PY'
import importlib
import torch
mods = [
    "torch",
    "transformers",
    "trl",
    "datasets",
    "accelerate",
    "lm_eval",
    "flash_attn",
    "einops",
]
for name in mods:
    importlib.import_module(name)
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM
print(f"ok  torch={torch.__version__} cuda={torch.version.cuda}")
print(f"ok  GKDConfig/GKDTrainer imported")
print(f"ok  flash_attn={importlib.import_module('flash_attn').__version__}")
if torch.cuda.is_available():
    x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    y = x @ x.t()
    print(f"ok  cuda gemm {tuple(y.shape)} on {torch.cuda.get_device_name(0)}")
else:
    print("skip cuda gemm (no GPU or driver < CUDA 12.9)")
PY

echo
echo "Done. Next:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "  python main_block_qat.py --help"
echo "  python main_e2e_distill.py --help"
echo
echo "Pack for cluster docker after this env is verified on a CUDA 12.9 node."
