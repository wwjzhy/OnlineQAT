# Qwen3-1.7B W3A16 实验记录

给**另一个集群上的 GPT / 人**看。不要改 `HOME`。脚本里的 `/zju_0038/...` 是旧机器默认路径，**新集群必须先 export 下面 3 个路径**。

**写法：** 按时间往下追加。每次新增实验要求，在文末开一节 `Exp #N`，写清「要什么 / 怎么跑 / 产出 / 状态」。**不要改已经写下的旧节。** 跑的时候按 `#1 → #2 → …` 顺序；后面的实验若依赖前面的产物，在该节写明。

公共约定（各 Exp 除非自己改，否则都遵守）：

- W3A16，不要 `--train-emb`，不要把 `--wbits` 改成 2
- 评测：evalscope 一套（`eval_paper_benchmarks.sh` 起 `vllm serve`，再走 openai_api），T=0.6，top_k=20，max_tokens=8192
  GSM8K、AIME24、AIME25、MATH-500、LiveCodeBench、MMLU-Redux、GPQA-Diamond、IFEval
- **每次 Stage 2（尤其 OPD）训完必做诊断落盘 + 出图**，不是可选项。详见文末「OPD 逐步诊断」。最低交付：
  1. 确认 `log/distill/<tag>/opd_step_metrics.jsonl` 存在且覆盖全程 step
  2. `merge_opd_timeline.py` 合并评测 → `output/plots/<tag>/opd_timeline.{csv,jsonl}`
  3. 同 step 轴画出至少：`truncation_rate`、`code_jump_rate`（+ amplification）、`grad_norm`、主评测分（GSM8K / MATH-500）；图存 `output/plots/<tag>/`
  4. 在该次 Exp 的**状态**里写上 metrics / timeline / 图路径
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

## Exp #3（2026-08-26 新增）— Stage 1 block QAT 也跑同一套评测

**要求：** Exp #1 的 Stage 1（`block_qat`）跑完后、**还没做 Stage 2 蒸馏**时，把该 checkpoint 转成 vLLM 格式，用公共约定那套 evalscope 评一遍。用来当 GKD（#1）和 OPD（#2）的蒸馏前基线。不要重训 Stage 1。

`eval_paper_benchmarks.sh` 吃的是标准 HF 权重，不能直接评 `output/block_qat/...`（fake-quant 模块）。先 `convert_to_hf_vllm_compatible_model.py`，产出目录加 `-blockqat`，**不要覆盖** `#1` 最终的 `output/vllm/Qwen3-1.7B-w3g128`。

**依赖：** `output/block_qat/Qwen3-1.7B-w3g128`（Exp #1 `--stage 1`）。没有就先跑那个。

**状态：** 未跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f output/block_qat/Qwen3-1.7B-w3g128/config.json

CUDA_VISIBLE_DEVICES="${CONVERT_GPU:-0}" python scripts/convert_to_hf_vllm_compatible_model.py \
  --base-id ./output/block_qat/Qwen3-1.7B-w3g128 \
  --save-dir ./output/vllm/Qwen3-1.7B-w3g128-blockqat \
  --wbits 3 \
  --group-size 128

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128-blockqat \
  ./output/eval/Qwen3-1.7B-w3g128-blockqat
```

和 `#1` / `#2` 的 eval 目录对比：

```
output/eval/Qwen3-1.7B-w3g128-blockqat   # 本实验：仅 Stage 1
output/eval/Qwen3-1.7B-w3g128            # #1：Stage 1 + GKD
output/eval/Qwen3-1.7B-w3g128-opd        # #2：Stage 1 + OPD
```

建议顺序：`#1 --stage 1` → **本实验评测** → `#1 --stage 2` 和 `#2`（可并行，都读同一份 Stage 1）。

---

## Exp #4（2026-08-28 新增）— 复现 ReasoningQAT 2-bit

**要求：** 和 `#1` 同一套流水线（Stage 1 block QAT + Stage 2 离线 GKD：gold completion 上 JSD + 0.2 CE + 转 vLLM + 公共约定那套评测），把 bit 换成 **W2A16**。这是脚本默认档，也是原论文 1.7B 的主设置。不要 `--train-emb`。本实验**覆盖**公共约定里的「不要把 `--wbits` 改成 2」。

和 `#1`（W3）不要混：自己训一份 Stage 1，产出目录是 `w2g128`，**不要覆盖** `w3g128`。

| | Exp #1 | Exp #4 |
|--|--|--|
| 位宽 | W3A16 | **W2A16** |
| Stage 1 `weight_lr` | `1e-5` | `2e-5` |
| Stage 2 lr / epoch | `1e-6` / 1（512 step） | `5e-6` / **3（1536 step）** |
| 产出 | `Qwen3-1.7B-w3g128` | `Qwen3-1.7B-w2g128` |

其余与 `#1` 相同：OpenThoughts 32768、batch 64、`max_length=8192`、`top_k=20`、评测 T=0.6 / 8192。本实验只做 GKD 复现，**不要**跑 OPD。

**依赖：** 无（不读 `#1` 的 Stage 1）。需要 `#1` 已经用过的环境、数据和 `MODEL_PATH`。

**状态：** 未跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

bash scripts/run_qwen3_1.7b.sh --wbits 2 --gpus 0,1,2,3,4,5,6,7
# 不是 8 卡：改 --gpus
# 40G OOM：再加 --max-length 4096

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w2g128 \
  ./output/eval/Qwen3-1.7B-w2g128
```

可拆成 `--stage 1` / `2` / `3`。冒烟：`LIMIT=1 bash scripts/eval_paper_benchmarks.sh ...`。

产出：

```
output/block_qat/Qwen3-1.7B-w2g128
output/distill/Qwen3-1.7B-w2g128
output/vllm/Qwen3-1.7B-w2g128
output/eval/Qwen3-1.7B-w2g128
```

和 `#1` 的 `output/eval/Qwen3-1.7B-w3g128` 对比。W2 的 Stage 2 是 3 epoch，比 W3 长约 3 倍。

---

## Exp #5（2026-09-01 新增）— W2 Stage 2 OPD，接 #4 的 Stage 1，每 5 step 存评测

**要求：** 不要重训 Stage 1。**直接读 Exp #4 Stage 1 训完的** `output/block_qat/Qwen3-1.7B-w2g128`，在这份 W2 checkpoint 上只跑 OPD **Stage 2**（student rollout + sampled reverse KL，不要 CE）。**最多 100 optimizer step**（不要跑满 W2 的 3 epoch / 1536 step）。**8 卡**训练。训练过程中 **每 5 个 step 存一份可转换 checkpoint**；8 卡占满训练，评测等 Stage 2 结束后再排队 convert + 公共约定那套 evalscope。不要 `--train-emb`。本实验覆盖公共约定里的「不要把 `--wbits` 改成 2」。

不要用 OPD 脚本的 `--stage all` / Stage 3：中间评测由 `eval_distill_checkpoints.sh` 做 convert，不写最终的 `output/vllm/Qwen3-1.7B-w2g128-opd`（避免和 #4 的 GKD `w2g128` 混）。产出后缀是 `w2g128-opd`。`#4` 的 Stage 2 GKD 可以同时跑，两边读同一份 Stage 1，写出目录不同。

| | Exp #2 | Exp #4 | Exp #5 |
|--|--|--|--|
| 位宽 | W3 | W2 GKD | **W2 OPD** |
| Stage 1 | `#1` 的 `w3g128` | **自己训 `w2g128`** | **不训，读 `#4` 这份** |
| Stage | 2+3 | 1+2+3 | **只 Stage 2，最多 100 step** |
| 中间 ckpt | 无 | 无 | **每 5 step** |
| 评测 | 训完再评 | 训完再评 | **每 5 step 存一份，训完 8 卡后排队评** |

本实验 **截断到 100 step**（`--max-steps 100`），每 5 step 约 **20 个评测点**（checkpoint-5 … checkpoint-100）再加一份 `final`。默认评完后删掉该 step 的 distill/vLLM 权重（只留 `output/eval/...`）；若要留权重：`KEEP_CHECKPOINTS=1`。

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`（**Exp #4 `--stage 1` 完成**）。没有就等 `#4` 的 Stage 1，不要在本实验里重跑 `run_qwen3_1.7b.sh --stage 1`。OPD 脚本不训 Stage 1；缺这份 checkpoint 时 `--stage 2` 会立刻退出。

**状态：** 未跑。脚本已接 `--max-steps` / `--save-steps` / `--eval-gpu`。把代码同步到新集群后，等 `#4` Stage 1 写完再跑下面。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

# 必须是 #4 Stage 1 的产物，不是另训一份
test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

# Stage 2 OPD：8 卡，每 5 step 存 ckpt，最多 100 step
# 40G OOM：再加 --max-length 4096
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 --max-steps 100 --save-steps 5

# 训练占满 8 卡，评测等结束后用 GPU 0 排队 convert+eval
bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/Qwen3-1.7B-w2g128-opd \
  --wbits 2 --eval-gpu 0
```

可选自检（不占 GPU）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/test_opd_stage2.py
```

看评测是否在跟：

```bash
tail -f log/eval/Qwen3-1.7B-w2g128-opd/watcher.log
ls output/eval/Qwen3-1.7B-w2g128-opd
```

产出：

```
output/block_qat/Qwen3-1.7B-w2g128          # 依赖：#4 Stage 1
output/distill/Qwen3-1.7B-w2g128-opd        # Stage 2 最终权重 + checkpoint-*（评完默认删中间点）
output/eval/Qwen3-1.7B-w2g128-opd/checkpoint-5
output/eval/Qwen3-1.7B-w2g128-opd/checkpoint-10
...
output/eval/Qwen3-1.7B-w2g128-opd/checkpoint-100
output/eval/Qwen3-1.7B-w2g128-opd/final     # step 100 结束后的整模评测
```

和 `#4` 的 `output/eval/Qwen3-1.7B-w2g128`（GKD）对比。不要覆盖 `#4`。

---

## Exp #6（2026-09-01 新增）— PV-OPD FullPair：W2 rollout + W4 精度验证

**要求：** 直接读取 Exp #4 Stage 1 的
`output/block_qat/Qwen3-1.7B-w2g128`，不要重训 Stage 1。W2 target
负责 rollout；冻结的 BF16 Teacher 和共享当前主权重、group、实数 clipping
range 的 W4 Probe 在同一条 W2 轨迹上打分。使用 W4 恢复方向/幅度生成
precision gate，加权现有 sampled reverse-KL。不要切换成 PPO ratio loss。

这是 **FullPair** 版本：更新 W2 的完整 master weights、非量化权重和
quantizer scale；`zero_point`、embedding、BF16 Teacher 与 W4 Probe 都不更新。
W4 只是 W2 量化器的临时 precision view，不单独保存 checkpoint。

**依赖：**

```
output/block_qat/Qwen3-1.7B-w2g128/config.json   # Exp #4 Stage 1
```

**状态：** 代码已实现，正式 8 卡实验未跑。

主实验（8 卡、effective batch 64、最多 100 optimizer step、每 5 step 存一次）：

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat
test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

bash scripts/run_qwen3_1.7b_pv_opd.sh \
  --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --probe-bits 4 \
  --max-steps 100 \
  --save-steps 5 \
  --gate-mode full
```

8 卡训练结束后，在 25/50/75/100 step 跑 Math/Code 主评测（保留 checkpoint，
以便失败后重跑）：

```bash
KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="aime24 aime25 math_500 live_code_bench" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/Qwen3-1.7B-w2g128-pv-opd \
  --wbits 2 --eval-gpu 0 \
  --steps 25,50,75,100 --skip-final

# final 转标准 HF/vLLM 后跑公共约定完整套件
bash scripts/run_qwen3_1.7b_pv_opd.sh --stage 3
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w2g128-pv-opd \
  ./output/eval/Qwen3-1.7B-w2g128-pv-opd/final
```

产出：

```
output/distill/Qwen3-1.7B-w2g128-pv-opd
output/eval/Qwen3-1.7B-w2g128-pv-opd/checkpoint-{25,50,75,100}
output/eval/Qwen3-1.7B-w2g128-pv-opd/final
log/distill/Qwen3-1.7B-w2g128-pv-opd
```

### 验证矩阵

所有方法必须使用相同 Stage 1、prompt 顺序、seed、8 卡、effective batch=64、
100 steps、rollout budget 和可训练参数。

1. Step 0：Exp #4 Stage 1 W2。
2. Standard OPD-FullPair：Exp #5，`g=1`。
3. PV-OPD-FullPair：本实验，`--gate-mode full`。
4. Sign-only 消融：本脚本加 `--gate-mode sign`，输出后缀 `-pv-opd-sign`。
5. Shuffled-gate 消融：本脚本加 `--gate-mode shuffled`，输出后缀
   `-pv-opd-shuffled`；保持 gate 稀疏率，检验收益是否只是稀疏正则化。

训练日志记录 gate keep-rate/均值、同号率、`|A_FP|`、`|A_prec|`、P99
adv clip、四段 token position 的 sampled-KL，以及数字/运算符/代码符号/普通
文本的 gate 均值。

每 5 step 保存 checkpoint 并检查上述固定诊断；公开 benchmark 建议只在
step 0/25/50/75/100 跑 AIME24/25、MATH-500、LiveCodeBench，最终模型再跑
公共约定的完整套件。判断成立要求：PV-OPD 在同一训练预算下稳定优于
Standard OPD，且优于 shuffled-gate；收益应主要出现在 Math/Code 长轨迹。

单测与短程 smoke：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/test_pv_opd.py
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/test_opd_stage2.py

# 正式 8 卡前，用短序列验证 DDP/显存/梯度；需要已有 Stage 1。
MAX_LENGTH=256 DATASET_SIZE=16 \
  bash scripts/run_qwen3_1.7b_pv_opd.sh \
  --stage 2 --gpus 0,1 --max-steps 1 --save-steps 1
```

---

## Exp #7（2026-09-04 新增）— W2-OPD，学习率降到 `1e-6`

**要求：** 和 Exp #5 同一设定（读 `#4` Stage 1 的 `w2g128`，student rollout + sampled reverse KL，CE=0，8 卡，effective batch 64，每 5 step 存 ckpt），**只改两处：学习率 `5e-6` → `1e-6`，最多 optimizer step `100` → `50`**。用来检验默认 W2-OPD 的评测振荡 / code-jump 是否主要是步长过大。不要 `--train-emb`，不要重训 Stage 1。

产出目录必须和 `#5` 分开：脚本在非默认 LR 时自动加后缀 `-lr1e-6`，写成 `Qwen3-1.7B-w2g128-opd-lr1e-6`，**不要覆盖** `#5` 的 `w2g128-opd`。

| | Exp #5 | Exp #7 |
|--|--|--|
| Stage 1 | `#4` 的 `w2g128` | **同一份** |
| Loss | sampled reverse KL | 相同 |
| LR | **`5e-6`（W2 默认）** | **`1e-6`** |
| max / save steps | **100 / 5** | **50 / 5** |
| 产出后缀 | `w2g128-opd` | `w2g128-opd-lr1e-6` |

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`（Exp #4 Stage 1）。没有就等 `#4`，不要在本实验重跑 Stage 1。

**状态：** 未跑。脚本已支持 `--lr`；非默认 LR 会改产出目录名。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

# 相对 #5：--lr 1e-6，且只跑 50 step
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 50 --save-steps 5 \
  --lr 1e-6

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/Qwen3-1.7B-w2g128-opd-lr1e-6 \
  --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,35,50 --skip-final
```

看训练时确认横幅是 `lr=1e-6`，产出是 `*-opd-lr1e-6` 而不是 `*-opd`。

产出：

```
output/distill/Qwen3-1.7B-w2g128-opd-lr1e-6
output/eval/Qwen3-1.7B-w2g128-opd-lr1e-6/checkpoint-{5,...,50}
log/distill/Qwen3-1.7B-w2g128-opd-lr1e-6
```

和 `#5`（`5e-6`）对比前 50 step：GSM8K 是否还出现 25→0.5 断崖、`grad_norm` 尖峰、code-jump ratio 是否明显下降。若 `1e-6` 稳住，则默认崩主要是步长问题，而不是 OPD 接线错误。

---

## Exp #8（2026-09-04 新增）— W2-OPD，主 LR `2e-6` + 30 step warmup（起始 `2e-7`）

**要求：** 仍读 Exp #4 Stage 1 的 `w2g128`，其余与 `#5/#7` 相同（student rollout + sampled reverse KL，CE=0，8 卡，effective batch 64，**最多 50 step**，每 5 step 存 ckpt）。相对默认 W2-OPD，改学习率日程：

| 项 | 默认 `#5` | 本实验 `#8` |
|--|--|--|
| 峰值 / 主文目标 LR | `5e-6` | **`2e-6`** |
| Warmup | `warmup_ratio=0.2`（约 10/50 step，从 0 升） | **固定前 30 optimizer step** |
| Warmup 起始 LR | 0 | **`2e-7`** |
| 产出后缀 | `w2g128-opd` | `w2g128-opd-lr2e-6-wu30-ws2e-7` |

不要 `--train-emb`，不要重训 Stage 1。用来对照「恒定小 LR（#7）」vs「略高峰值但长 warmup、非零起点（本实验）」对 code-jump / 评测振荡的影响。

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`（Exp #4 Stage 1）。

**状态：** 未跑。脚本已支持 `--warmup-steps` / `--warmup-start-lr`。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 50 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7 \
  --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,35,50 --skip-final
```

训练横幅应看到 `lr=2e-6`、`warmup_steps=30`、`warmup_start_lr=2e-7`。注意：50 step 里有 30 step 在 warmup，峰值 LR 段只有约 20 step。

产出：

```
output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7
output/eval/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7/checkpoint-{5,...,50}
log/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7
```

和 `#5`（`5e-6`）/ `#7`（`1e-6`）对比 GSM8K 曲线、`grad_norm`、code-jump。

---

## Exp #9（2026-09-06 新增）— ReasoningQAT：完整复现论文主方法

**要求：** 按 ReasoningQAT 论文目标函数做 Stage 1+2+3，**不要**用 Exp #1 的 plain CE + JSD 脚本当「论文复现」。Stage 1 与 `#1` 相同（`sweep_0.8` block QAT）；Stage 2 换成论文的 teacher-guided reward rectification + forward KL + cosine。产出后缀 `-reasoningqat`，**不要覆盖** `#1` 的 `w3g128` distill。

| | Exp #1（旧「GKD」脚本） | **本实验 ReasoningQAT** |
|--|--|--|
| 脚本 | `run_qwen3_1.7b.sh` | **`run_qwen3_1.7b_reasoningqat.sh`** |
| CE | 普通 `student_outputs.loss` | **`--use_teacher_weight`**：\(L_t=\mathrm{CE}\cdot\mathrm{sg}(\pi_T(y^\star))\) |
| KD | `kd_loss_type=jsd`（且曾被 `beta=0.5` 盖成对称 JSD） | **`forward_kl`，`gkd_beta=1.0`，`top_k=20`** |
| 调度 | linear + `warmup_ratio=0.2` | **`lr_scheduler_type=cosine`** + warmup_ratio 0.2 |
| 权重 | \(\alpha=0.2\), \(\beta=1.0\) | 相同 |
| 产出 | `Qwen3-1.7B-w3g128` | **`Qwen3-1.7B-w3g128-reasoningqat`** |

W3 默认：`weight_lr=1e-5`，Stage 2 `lr=1e-6`，1 epoch，batch 64，OpenThoughts 32768，`max_length=8192`，不 `--train-emb`。W2 用 `--wbits 2`（Stage 1 `2e-5`，Stage 2 `5e-6` / 3 epoch）。

论文公式：\(L=0.2\,L_t+1.0\,\mathrm{KL}(\pi_T\Vert\pi_S)\)。主表看 MATH-500 / LiveCodeBench / MMLU-Redux / GPQA-Diamond / IFEval 五任务均值（W3 论文约 **55.2**）；GSM8K/AIME 可顺带评，但不在论文主 avg 里。

**依赖：** 无。若 `#1` 已有 `output/block_qat/Qwen3-1.7B-w3g128`，可 `--stage 2` 直接复用 Stage 1。

**状态：** 未跑。已修 `PolicyGKDTrainer` 不再把 `beta` 硬盖成 0.5；新增 `--lr_scheduler_type` / `--gkd_beta`。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

# 整条（Stage 1 没有则训；有则用 --skip-existing 跳过 Stage 1）
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --gpus 0,1,2,3,4,5,6,7

# 或只跑 Stage 2+3（读已有 Stage 1）
test -f output/block_qat/Qwen3-1.7B-w3g128/config.json
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --stage 2 --gpus 0,1,2,3,4,5,6,7
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --stage 3

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w3g128-reasoningqat \
  ./output/eval/Qwen3-1.7B-w3g128-reasoningqat
```

训练横幅应看到 `loss: 0.2 * teacher-weighted CE + 1.0 * forward_kl`，产出目录带 `-reasoningqat`。

产出：

```
output/block_qat/Qwen3-1.7B-w3g128          # 可与 #1 共用
output/distill/Qwen3-1.7B-w3g128-reasoningqat
output/vllm/Qwen3-1.7B-w3g128-reasoningqat
output/eval/Qwen3-1.7B-w3g128-reasoningqat
```

和 `#1` 的 `output/eval/Qwen3-1.7B-w3g128` 对比：本实验应更接近论文主表。

---

## Exp #10（2026-09-06 新增）— W2-OPD，30 step warmup 后 **恒定峰值 LR**，共 100 step

**要求：** 读 Exp #4 Stage 1 的 `w2g128`，OPD 设定与 `#8` 相同（student rollout + sampled reverse KL，CE=0，8 卡，batch 64，每 5 step 存 ckpt），学习率日程改为：

| 项 | `#8` | **本实验 `#10`** |
|--|--|--|
| 峰值 LR | `2e-6` | **相同** |
| Warmup | 前 30 step，自 `2e-7` 升到峰值 | **相同** |
| Warmup 之后 | linear **衰减到 0**（50 step 里只剩 ~20 step 峰值且在掉） | **`constant_with_warmup`：峰值保持不变（stable）** |
| 总 step | 50 | **100**（30 warmup + **70 stable**） |
| 产出后缀 | `...-wu30-ws2e-7` | `...-wu30-ws2e-7-schhold` |

不要 `--train-emb`，不要重训 Stage 1。用来检验：在 `#8` 已验证的「长 warmup + `2e-6`」上，**拉长稳定段、且不衰减**，GSM/MATH 是否继续涨、会不会再次崩。

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`（Exp #4 Stage 1）。

**状态：** 未跑。需 `--lr-scheduler constant_with_warmup`（脚本 tag 缩写 `schhold`）。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 100 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold \
  --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,35,50,75,100 --skip-final
```

横幅应有 `lr_scheduler=constant_with_warmup`、`max_steps=100`。产出不要覆盖 `#8`。

产出：

```
output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
output/eval/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-{5,...,100}
log/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
```

和 `#8`（50 step + 衰减）对比：stable 段（step 30–100）是否单调更好、有无二次崩溃。

---

## 续训 / Resume（2026-09-06）

`--save-steps N` 现在会同时写两套东西：

| 路径 | 内容 | 能否 `--resume` |
|--|--|--|
| `log/distill/<tag>/checkpoint-*` | HF Trainer 全量（权重+Adam+scheduler） | **能** |
| `output/distill/<tag>/checkpoint-*` | 仅评测用假量化快照 | **不能**（用 `--init-from`） |

**真续训（推荐，需本次改动之后新跑过的实验）：** 把 `--max-steps` 提高到大于当前 `global_step`。

```bash
# 例：#8 已训到 50，接着训到 100（同一 tag，接上 Adam）
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 100 --save-steps 5 \
  --lr 2e-6 --warmup-steps 30 --warmup-start-lr 2e-7 \
  --resume
```

**旧 #7/#8（只有 `output/distill/.../checkpoint-50`）：** 没有 Trainer ckpt，只能权重热启（Adam 清零；建议 `--warmup-steps 0` + 新 suffix）：

```bash
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 50 --save-steps 5 \
  --lr 2e-6 --warmup-steps 0 \
  --init-from ./output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7/checkpoint-50 \
  --run-suffix cont50
```

---

## OPD 逐步诊断（truncation / 跳码 / grad_norm / eval）

**强制：** 每一次 OPD / Stage 2 训练结束后，都要把下面这套 log **收齐、合并、画图**，再在该 Exp 的状态里挂路径。只报最终 GSM 分、不交 timeline 图，算没做完。

### 训练时自动落盘

Stage 2 开 `--opd` 时，rank0 自动写：

| 文件 | 内容 |
|------|------|
| `log/distill/<tag>/opd_step_metrics.jsonl` | 每 optimizer step 一行 |

字段（前缀 `opd/` 的也进 Trainer log）：

- **truncation：** `truncation_rate`、`eos_rate`、`mean_response_len`、`hit_max_length_rate`（无 EOS 且顶满 `max_length` 算截断）
- **跳码：** `code_jump_rate`、`code_jump_count`、`master_rel_change`、`code_jump_amplification`
- **优化：** `loss`、`grad_norm`、`learning_rate`

训完先确认 JSONL 行数 ≈ `max_steps`（resume 则覆盖续训区间）。缺文件 = 诊断没开，不要直接评测结案。

### 训后必做：合并 + 出图

```bash
# <tag> 换成该次 DISTILL_TAG，例如 Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
python scripts/merge_opd_timeline.py \
  --metrics log/distill/<tag>/opd_step_metrics.jsonl \
  --eval-root output/eval/<tag> \
  --out-dir output/plots/<tag>
```

产出：

- `output/plots/<tag>/opd_timeline.csv`
- `output/plots/<tag>/opd_timeline.jsonl`

**同一 step 轴至少画 4 条曲线（可多子图）：**

1. `opd/truncation_rate`（可叠 `eos_rate` / `mean_response_len`）
2. `opd/code_jump_rate` + `opd/code_jump_amplification`
3. `grad_norm`（可叠 `learning_rate`）
4. eval：`GSM8K` / `MATH-500`（以及该次 `--save-steps` 有的 checkpoint 分）

图保存到 `output/plots/<tag>/`（如 `timeline.png` / 分面板 PDF）。状态栏示例：

```text
metrics: log/distill/<tag>/opd_step_metrics.jsonl
timeline: output/plots/<tag>/opd_timeline.csv
plots: output/plots/<tag>/timeline.png
```

---

## 以后怎么加

下一次加实验：复制下面模板接到文末，编号 +1。不要回头改 `#1`、`#2`、`#3` 的要求。

```md
## Exp #N（YYYY-MM-DD 新增）— 一句话目标

**要求：** …

**依赖：** …（没有就写「无」）

**状态：** 未跑 / 跑到哪 / 结果路径
（OPD/Stage2 跑完须附：opd_step_metrics.jsonl、opd_timeline、plots 路径）

```bash
# 只写这一次的命令
# 训完：merge_opd_timeline.py + 出图 → output/plots/<tag>/
```
```
