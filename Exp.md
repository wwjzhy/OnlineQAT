# Qwen3-1.7B W3A16 复现（ReasoningQAT）

给**另一个集群上的 GPT / 人**看：clone 本仓库后，先填 3 个本机路径，再按顺序跑。不要改 `HOME`。

目标：**W3A16**（3-bit 权重量化）、**不训练 embedding**（不要加 `--train-emb`）。

评测只用 **evalscope 一套**（论文采样：T=0.6，top_k=20，max_tokens=8192）：

GSM8K、AIME24、AIME25、MATH-500、LiveCodeBench、MMLU-Redux、GPQA-Diamond、IFEval

硬件：**8 张同构 GPU**（推荐 H20 96G）。A100 40G 若 Stage 2 OOM，训练命令加 `--max-length 4096`。不要混用 A100 和 H20。CUDA 建议 12.9（`scripts/setup_env_cu129.sh`）。

脚本里仍有本实验室默认路径（`/zju_0038/...`）。**新集群必须 export 下面变量，否则会去找不存在的路径。**

---

## 先填这 3 个路径（新集群必做）

`HOME` 不用设。下面三个必须是**当前机器上真实存在的目录**。

| 变量 | 是什么 | 怎么查 | 例子（不要照抄） |
|------|--------|--------|------------------|
| `CONDA_ROOT` | conda 安装根目录，下面要有 `bin/conda` 和 `etc/profile.d/conda.sh` | `dirname $(dirname $(which conda))`；若还没装 conda，先装到 `$HOME/miniconda3` | `/opt/conda` 或 `$HOME/miniconda3` |
| `MODEL_PATH` | Qwen3-1.7B 权重目录，里面必须有 `config.json` | 集群模型盘上找 `Qwen3-1.7B`；没有就从 Hugging Face `Qwen/Qwen3-1.7B` 下到本地 | `/data/models/Qwen3-1.7B` |
| `HF_HOME` | Hugging Face 缓存（OpenThoughts 等） | 固定写成仓库内 `$PWD/hf_cache`，不要用家目录 | `$PWD/hf_cache` |

另外两个一起设（不是新路径）：

- `TEACHER_MODEL=$MODEL_PATH`（Stage 2 教师 = 同一份基座）
- `HF_ENDPOINT=https://hf-mirror.com`（国内；集群能直连 huggingface.co 可改或不设）
- `HF_TOKEN`：仅训练数据 OpenThoughts 若要求登录才需要。评测走 ModelScope，**GPQA-Diamond 不需要 HF 审核/token**

GPU：先 `nvidia-smi -L` 看有几张。8 张用 `--gpus 0,1,2,3,4,5,6,7`。不是 8 张就改成实际 id，且 GPU 数必须能整除 64（1/2/4/8 都可以）。

在仓库根目录执行一次（把两处 `/改成你的路径` 换成上面查到的值）：

```bash
cd OnlineQAT   # 已在仓库根则可省略

export CONDA_ROOT=/改成你的conda根目录          # 仅当 conda 就在 $HOME/miniconda3 时可写成 $HOME/miniconda3
export MODEL_PATH=/改成你的/Qwen3-1.7B         # 目录内必须有 config.json
export TEACHER_MODEL=$MODEL_PATH
export HF_HOME=$PWD/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export ENV_NAME=reasoningqat
export PIP_CONFIG_FILE=/dev/null               # 避免集群坏掉的 pip.conf
```

检查：

```bash
test -x "$CONDA_ROOT/bin/conda" && echo "conda ok"
test -f "$MODEL_PATH/config.json" && echo "model ok"
nvidia-smi -L
```

`conda ok` 和 `model ok` 都打印出来再往下。后面每开一个新 shell 都要重新 export 这几行（或写进 `~/.bashrc`）。

---

## 0. 环境

```bash
git clone https://github.com/wwjzhy/OnlineQAT.git OnlineQAT
cd OnlineQAT
# 先做完上面的 export

bash scripts/setup_env_cu129.sh
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
pip install -r requirements-eval.txt
```

`setup_env_cu129.sh` 会装 torch cu129、transformers、**flash-attn**（Stage1/2 需要，编译可能较久）。不要用仓库根目录的 `requirements.txt`（那是旧 CUDA 的 pin）。

---

## 1. 下载训练数据

```bash
cd OnlineQAT
# 确认 HF_HOME / HF_ENDPOINT 仍在当前 shell
python scripts/download_datasets.py
```

评测集由 evalscope 在第 3 步首次运行时拉取，不必单独下。

| 用途 | 路径 |
|------|------|
| Stage 1/2 OpenThoughts | `HF_HOME` 缓存 |
| Stage 1 FineWeb 子集 | `data/raw/fineweb_edu_subset.jsonl` |

---

## 2. 训练 W3A16

```bash
cd OnlineQAT
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
# 确认 CONDA_ROOT / MODEL_PATH / TEACHER_MODEL / HF_HOME / HF_ENDPOINT 仍在当前 shell

bash scripts/run_qwen3_1.7b.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7
# 不是 8 卡：改 --gpus 为实际 id，例如 4 卡 --gpus 0,1,2,3
# 40G OOM:  再加 --max-length 4096
```

必须 `--wbits 3`（脚本默认仍是 2）。不要加 `--train-emb`。产出：

```
output/block_qat/Qwen3-1.7B-w3g128
output/distill/Qwen3-1.7B-w3g128
output/vllm/Qwen3-1.7B-w3g128
```

---

## 3. 评测（一套）

Stage 3 完成后，单卡：

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128 \
  ./output/eval/Qwen3-1.7B-w3g128
```

结果在 `output/eval/Qwen3-1.7B-w3g128`。冒烟（每集 1 条）：`LIMIT=1 bash scripts/eval_paper_benchmarks.sh ...`

加长生成：

```bash
MAX_TOKENS=16384 bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128 \
  ./output/eval/Qwen3-1.7B-w3g128-16k
```

评测集从 **ModelScope** 拉取，**不需要** Hugging Face 的 GPQA 审核/`HF_TOKEN`。LiveCodeBench 只用 `release_latest` 一个 subset（evalscope 默认会下全部版本，体积很大）。IFEval 依赖 `evalscope[ifeval]`（`requirements-eval.txt` 已包含）。

---

## 4. 一条龙（填路径后整段复制）

把前两行路径改成本机值，其余不要改。

```bash
set -euo pipefail
cd OnlineQAT

export CONDA_ROOT=/改成你的conda根目录
export MODEL_PATH=/改成你的/Qwen3-1.7B
export TEACHER_MODEL=$MODEL_PATH
export HF_HOME=$PWD/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export ENV_NAME=reasoningqat
export PIP_CONFIG_FILE=/dev/null

test -x "$CONDA_ROOT/bin/conda"
test -f "$MODEL_PATH/config.json"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
pip install -r requirements-eval.txt

python scripts/download_datasets.py
bash scripts/run_qwen3_1.7b.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128 \
  ./output/eval/Qwen3-1.7B-w3g128
```

首次在新机器还要先跑一次 `bash scripts/setup_env_cu129.sh`（第 0 步），再执行这一段。环境已经装好可跳过 setup。

---

## GPT 在新集群上怎么跑

1. 读完本文，**不要**使用任何 `/zju_0038` 路径。
2. 查清 `CONDA_ROOT`、`MODEL_PATH`、GPU 列表，按「先填这 3 个路径」export。
3. 按 0 → 1 → 2 → 3 顺序执行；或环境已装好则跑第 4 节一条龙。
4. 缺 conda / 缺模型 / GPU 不足 / 下载失败时停下来问人，不要用错误默认路径硬跑。
5. 不要加 `--train-emb`，不要把 `--wbits` 改成 2。
