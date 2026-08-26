# Qwen3-1.7B W3A16 实验记录

给**另一个集群上的 GPT / 人**看。不要改 `HOME`。脚本里的 `/zju_0038/...` 是旧机器默认路径，**新集群必须先 export 下面 3 个路径**。

**写法：** 按时间往下追加。每次新增实验要求，在文末开一节 `Exp #N`，写清「要什么 / 怎么跑 / 产出 / 状态」。**不要改已经写下的旧节。** 跑的时候按 `#1 → #2 → …` 顺序；后面的实验若依赖前面的产物，在该节写明。

公共约定（各 Exp 除非自己改，否则都遵守）：

- W3A16，不要 `--train-emb`，不要把 `--wbits` 改成 2
- 评测：evalscope 一套，T=0.6，top_k=20，max_tokens=8192
  GSM8K、AIME24、AIME25、MATH-500、LiveCodeBench、MMLU-Redux、GPQA-Diamond、IFEval
- 硬件：8 张同构 GPU（推荐 H20 96G）。A100 40G Stage 2 OOM 则该次命令加 `--max-length 4096`。不要混用 A100 和 H20
- CUDA：`scripts/setup_env_cu129.sh`

---

## 机器准备（不是实验；每台新机器做一次）

`HOME` 不用设。三个变量必须是**当前机器上真实存在的目录**。

| 变量 | 是什么 | 怎么查 | 例子（不要照抄） |
|------|--------|--------|------------------|
| `CONDA_ROOT` | conda 根目录，下面有 `bin/conda` 和 `etc/profile.d/conda.sh` | `dirname $(dirname $(which conda))` | `/opt/conda` 或 `$HOME/miniconda3` |
| `MODEL_PATH` | Qwen3-1.7B，目录内有 `config.json` | 集群模型盘；没有就下 `Qwen/Qwen3-1.7B` | `/data/models/Qwen3-1.7B` |
| `HF_HOME` | Hugging Face 缓存 | 写成仓库内 `$PWD/hf_cache` | `$PWD/hf_cache` |

一起设：`TEACHER_MODEL=$MODEL_PATH`；国内 `HF_ENDPOINT=https://hf-mirror.com`。评测走 ModelScope，**GPQA 不需要 HF token**。OpenThoughts 若要登录才设 `HF_TOKEN`。

GPU：`nvidia-smi -L`。8 张用 `--gpus 0,1,2,3,4,5,6,7`。张数必须能整除 64（1/2/4/8）。

```bash
cd OnlineQAT   # 已在仓库根则可省略

export CONDA_ROOT=/改成你的conda根目录
export MODEL_PATH=/改成你的/Qwen3-1.7B
export TEACHER_MODEL=$MODEL_PATH
export HF_HOME=$PWD/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export ENV_NAME=reasoningqat
export PIP_CONFIG_FILE=/dev/null

test -x "$CONDA_ROOT/bin/conda" && echo "conda ok"
test -f "$MODEL_PATH/config.json" && echo "model ok"
nvidia-smi -L
```

环境：

```bash
git clone https://github.com/wwjzhy/OnlineQAT.git OnlineQAT
cd OnlineQAT
# 先做完上面的 export
bash scripts/setup_env_cu129.sh
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
pip install -r requirements-eval.txt
```

训练数据：

```bash
python scripts/download_datasets.py
```

| 用途 | 路径 |
|------|------|
| Stage 1/2 OpenThoughts | `HF_HOME` 缓存 |
| Stage 1 FineWeb 子集 | `data/raw/fineweb_edu_subset.jsonl` |

评测集在首次 `eval_paper_benchmarks.sh` 时从 ModelScope 拉。LiveCodeBench 只用 `release_latest`。IFEval 见 `requirements-eval.txt`。

每开一个新 shell 都要重新 export（或写进 `~/.bashrc`）。缺 conda / 缺模型 / GPU 不够就停下来问人。

---

## Exp #1（原先）— 复现 ReasoningQAT GKD

**要求：** Qwen3-1.7B W3A16，论文 Stage 1 block QAT + Stage 2 离线蒸馏（数据集 gold completion 上 JSD + 0.2 CE）+ 转 vLLM + 上面那套评测。不训练 embedding。

**状态（旧集群 `/zju_0038`）：** 环境、数据、BF16 基座 `LIMIT=1` 冒烟做过。**W3 没训完**，没有 `output/block_qat/Qwen3-1.7B-w3g128`。新集群要整条重跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

bash scripts/run_qwen3_1.7b.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7
# 不是 8 卡：改 --gpus，例如 --gpus 0,1,2,3
# 40G OOM：再加 --max-length 4096

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128 \
  ./output/eval/Qwen3-1.7B-w3g128
```

可拆成 `--stage 1` / `2` / `3`。冒烟：`LIMIT=1 bash scripts/eval_paper_benchmarks.sh ...`（分数无意义）。加长：`MAX_TOKENS=16384 ... ./output/eval/Qwen3-1.7B-w3g128-16k`。

产出：

```
output/block_qat/Qwen3-1.7B-w3g128
output/distill/Qwen3-1.7B-w3g128
output/vllm/Qwen3-1.7B-w3g128
output/eval/Qwen3-1.7B-w3g128
```

---

## Exp #2（2026-08-26 新增）— 纯 OPD，和 #1 的 GKD 对比

**要求：** 在 **同一份 Exp #1 Stage 1**、同一套数据与超参上，把 Stage 2 换成 on-policy：student 自己 rollout，只用 JSD，**不要** CE / gold SFT。单独脚本，产出不要覆盖 #1。

和 #1 相同：OpenThoughts 32768、lr / epoch / batch 64、`max_length=8192`、`top_k=20`、不训练 embedding。只改 Stage 2 序列来源和 loss。

| | Exp #1 GKD | Exp #2 OPD |
|--|--|--|
| 脚本 | `scripts/run_qwen3_1.7b.sh` | `scripts/run_qwen3_1.7b_opd.sh` |
| 序列 | gold completion | student rollout（T=0.6，总长 cap 8192） |
| Loss | `0.2 * CE(gold) + 1.0 * JSD` | `1.0 * JSD`（`--cross_entropy_weight 0`） |
| 产出后缀 | `Qwen3-1.7B-w3g128` | `Qwen3-1.7B-w3g128-opd` |

**依赖：** `output/block_qat/Qwen3-1.7B-w3g128`（Exp #1 Stage 1）。没有就先跑 `#1` 的 `--stage 1`。OPD 脚本不跑、也不要重训 Stage 1。

**状态：** 代码和 CPU 接线测试已加（`tests/test_opd_stage2.py`）。训练和评测还没跑。

可选自检（不占 GPU）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/test_opd_stage2.py
```

```bash
# 确认 #1 的 Stage 1 已在
test -f output/block_qat/Qwen3-1.7B-w3g128/config.json

bash scripts/run_qwen3_1.7b_opd.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128-opd \
  ./output/eval/Qwen3-1.7B-w3g128-opd
```

产出：`output/{distill,vllm,eval}/Qwen3-1.7B-w3g128-opd`。和 `#1` 的 eval 目录对比。

---

## 以后怎么加

下一次加实验：复制下面模板接到文末，编号 +1。不要回头改 `#1`、`#2` 的要求。

```md
## Exp #N（YYYY-MM-DD 新增）— 一句话目标

**要求：** …

**依赖：** …（没有就写「无」）

**状态：** 未跑 / 跑到哪 / 结果路径

```bash
# 只写这一次的命令
```
```
