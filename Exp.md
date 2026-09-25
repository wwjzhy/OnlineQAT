# Qwen3-1.7B W3A16 实验记录

给**另一个集群上的 GPT / 人**看。不要改 `HOME`。脚本里的 `/zju_0038/...` 是旧机器默认路径，**新集群必须先 export 下面 3 个路径**。

**写法：** 按时间往下追加。每次新增实验要求，在文末开一节 `Exp #N`，写清「要什么 / 怎么跑 / 产出 / 状态」。**不要改已经写下的旧节。** 跑的时候按 `#1 → #2 → …` 顺序；后面的实验若依赖前面的产物，在该节写明。

公共约定（各 Exp 除非自己改，否则都遵守）：

- W3A16，不要 `--train-emb`，不要把 `--wbits` 改成 2
- 评测：evalscope 一套（`eval_paper_benchmarks.sh` 起 `vllm serve`，再走 openai_api），T=0.6，top_k=20，max_tokens=8192
  GSM8K、AIME24、AIME25、MATH-500、LiveCodeBench、MMLU-Redux、GPQA-Diamond、IFEval
- **结果报告不只交 benchmark 分。** 每次 Stage 2 / OPD 结案，状态里必须同时写清 **时间与吞吐**（见文末「结果报告清单」），缺训练时间 / rollout 时间算没做完
- **每次 Stage 2（尤其 OPD）训完必做诊断落盘 + 出图**，不是可选项。详见文末「OPD 逐步诊断」。最低交付：
  1. 确认 `log/distill/<tag>/opd_step_metrics.jsonl` 存在且覆盖全程 step
  2. `merge_opd_timeline.py` 合并评测 → `output/plots/<tag>/opd_timeline.{csv,jsonl}`
  3. 同 step 轴画出至少：`truncation_rate`、`code_jump_rate`（+ amplification）、`grad_norm`、主评测分（GSM8K / MATH-500）；图存 `output/plots/<tag>/`
  4. 在该次 Exp 的**状态**里写上 metrics / timeline / 图路径，以及下面「结果报告清单」里的时间项
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

## Exp #11（2026-09-07 新增）— ReasoningQAT 论文主方法跑 **W2**

**要求：** 用和 `#9` 同一套**论文目标函数**在 **W2A16** 上跑满 Stage 1+2+3 + 公共约定评测。**不要**用 Exp #4 的 `run_qwen3_1.7b.sh`（那是旧 GKD：plain CE + JSD）当「论文 W2 结果」。本实验覆盖公共约定里的「不要把 `--wbits` 改成 2」。不要 `--train-emb`，不要跑 OPD。

| | Exp #4 | Exp #9 | **本实验 `#11`** |
|--|--|--|--|
| 位宽 | W2 | W3 | **W2** |
| 脚本 | `run_qwen3_1.7b.sh` | `run_qwen3_1.7b_reasoningqat.sh` | **同 `#9` 脚本** |
| Stage 2 loss | plain CE + JSD | teacher-weighted CE + `forward_kl` + cosine | **同 `#9`** |
| Stage 1 `weight_lr` | `2e-5` | `1e-5` | **`2e-5`** |
| Stage 2 lr / epoch | `5e-6` / 3（1536 step） | `1e-6` / 1 | **`5e-6` / 3（1536 step）** |
| 产出 | `…-w2g128` | `…-w3g128-reasoningqat` | **`…-w2g128-reasoningqat`** |

论文公式不变：\(L=0.2\,L_t+1.0\,\mathrm{KL}(\pi_T\Vert\pi_S)\)，`top_k=20`，`gkd_beta=1.0`，cosine + warmup_ratio 0.2。主表看 MATH-500 / LiveCodeBench / MMLU-Redux / GPQA-Diamond / IFEval 五任务均值；GSM8K/AIME 顺带评。和 `#4`（同 bit、错 loss）、`#9`（同 loss、W3）对比。

**依赖：** 无。若 `#4` 已有 `output/block_qat/Qwen3-1.7B-w2g128`，可 `--stage 2` 复用 Stage 1（**不要**覆盖 `#4` 的 distill `w2g128`；本实验写出 `-reasoningqat`）。

**状态：** 未跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

# 整条（无 Stage 1 则训；有 #4 Stage 1 可用 --skip-existing 跳过 Stage 1）
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 2 --gpus 0,1,2,3,4,5,6,7

# 或只跑 Stage 2+3
test -f output/block_qat/Qwen3-1.7B-w2g128/config.json
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 2 --stage 2 --gpus 0,1,2,3,4,5,6,7
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 2 --stage 3

bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-1.7B-w2g128-reasoningqat \
  ./output/eval/Qwen3-1.7B-w2g128-reasoningqat
```

横幅应看到 `loss: 0.2 * teacher-weighted CE + 1.0 * forward_kl`、`wbits=2`、`lr=5e-6`、`epochs=3`。W2 Stage 2 约 1536 step，比 `#9` 长约 3 倍。

产出：

```
output/block_qat/Qwen3-1.7B-w2g128                 # 可与 #4 共用
output/distill/Qwen3-1.7B-w2g128-reasoningqat
output/vllm/Qwen3-1.7B-w2g128-reasoningqat
output/eval/Qwen3-1.7B-w2g128-reasoningqat
```

对比：`#4` 的 `output/eval/Qwen3-1.7B-w2g128`（旧 GKD）、`#9` 的 `…-w3g128-reasoningqat`（论文方法 W3）。不要覆盖 `#4` / `#9`。

---

## Exp #12（2026-09-09 新增）— 汇总 W2/W3 Stage1 + KD + ReasoningQAT 的**训练时间**与**是否收敛**

**要求：** 不重训。对下面 6 档**已跑完或正在跑的**实验，从 log / `trainer_state.json` 抽出墙钟与 loss，填表；判断每档是否看起来收敛。硬件一行写清（如 8×H20）。本实验是**事后统计**，不是新训练。

| 档 | 对应 Exp | 日志 / 产出 tag |
|--|--|--|
| **W3-Stg1** | `#1` / `#9` 共用 Stage 1 | `log/block_qat/Qwen3-1.7B-w3g128`，`output/block_qat/Qwen3-1.7B-w3g128` |
| **W3-KD** | `#1` 离线 GKD（plain CE+JSD） | `log/distill/Qwen3-1.7B-w3g128` |
| **W3-ReasoningQAT** | `#9` | `log/distill/Qwen3-1.7B-w3g128-reasoningqat` |
| **W2-Stg1** | `#4` / `#11` 共用 Stage 1 | `log/block_qat/Qwen3-1.7B-w2g128` |
| **W2-KD** | `#4` 离线 GKD | `log/distill/Qwen3-1.7B-w2g128` |
| **W2-ReasoningQAT** | `#11` | `log/distill/Qwen3-1.7B-w2g128-reasoningqat` |

### 时间怎么读

**Stage 2（KD / ReasoningQAT）：**

```bash
# train_runtime（秒）+ 最终 loss
python - <<'PY'
import json
from pathlib import Path
for tag in [
  "Qwen3-1.7B-w3g128",
  "Qwen3-1.7B-w3g128-reasoningqat",
  "Qwen3-1.7B-w2g128",
  "Qwen3-1.7B-w2g128-reasoningqat",
]:
    p = Path(f"log/distill/{tag}/trainer_state.json")
    if not p.exists():
        # resume 时也可能在 checkpoint-* 下
        cands = sorted(Path(f"log/distill/{tag}").glob("checkpoint-*/trainer_state.json"))
        p = cands[-1] if cands else None
    print("====", tag)
    if not p or not p.exists():
        print("MISSING"); continue
    st = json.loads(p.read_text())
    hist = st.get("log_history", [])
    runtime = next((h.get("train_runtime") for h in reversed(hist) if "train_runtime" in h), None)
    losses = [h["loss"] for h in hist if "loss" in h]
    print("global_step", st.get("global_step"), "train_runtime_s", runtime,
          "hours", None if runtime is None else round(runtime/3600, 2))
    if losses:
        n = max(1, len(losses)//10)
        print("loss_first_avg", round(sum(losses[:n])/n, 4),
              "loss_last_avg", round(sum(losses[-n:])/n, 4),
              "loss_min", round(min(losses), 4), "n_logged", len(losses))
PY
```

也可用训练 shell 起止时间 / `log/distill/<tag>/*.log` 时间戳交叉核对。填表时写：**墙钟小时、GPU 数、总 step（W3 KD/RQ≈512；W2 KD/RQ≈1536）、sec/step**。

**Stage 1（block QAT）：** 看 `log/block_qat/<tag>/` 里 logger 起止、或脚本打印的 quantization wall time；没有 `train_runtime` 字段时用日志时间戳差。注明 **1 GPU**（Stage 1 不能 8 卡）。

### 收敛能不能从 loss 看出来？

**能看个大概，但不能单靠 loss 定论。**

| | 能看什么 | 局限 |
|--|--|--|
| Stage 2 loss 曲线 | 前半是否下降、后半是否走平、有无后期炸/震荡 | 不同方法 loss 尺度不同（JSD vs forward_kl），**不能跨方法比绝对 loss**；cosine 末期会再掉一点，不等于「没收敛」 |
| Stage 1 | 按层 recon / block loss 是否下降 | 是 block-wise，和 Stage 2 不可比 |
| 更可靠 | 同设定下 eval（GSM/MATH 等）随 step 是否稳住 | loss 降但评测掉 = 过拟合/分布漂，仍算「优化收敛、任务未稳」 |

**判定口径（写入状态）：**

1. **看起来收敛：** 后 20% step 的平均 loss ≤ 前 20% 的平均 loss，且后半无持续上升/剧烈尖峰  
2. **可疑 / 未收敛：** 后期 loss 回升、NaN、或中途挂掉未跑满 step  
3. **仅 loss 不足：** 若只有 final loss 没有曲线，标「无法判断」，去补 `log_history` 或 eval

填表模板（状态里贴齐）：

| 档 | 墙钟 | GPU | steps | sec/step | loss 前→后 | 收敛判定 |
|--|--|--|--|--|--|--|
| W3-Stg1 | | 1 | | | | |
| W3-KD | | 8 | ~512 | | | |
| W3-ReasoningQAT | | 8 | ~512 | | | |
| W2-Stg1 | | 1 | | | | |
| W2-KD | | 8 | ~1536 | | | |
| W2-ReasoningQAT | | 8 | ~1536 | | | |

**依赖：** 上表 6 档在**训练集群**上的 `log/`（本机 `/zju_0038` 可能只有空壳，以实际跑完的机器为准）。

**状态：** 未填。本仓库当前几乎无 `trainer_state.json`；到有完整 `log/distill` / `log/block_qat` 的机器上跑上面的脚本后，把表填回本节。

产出：

```
output/plots/train_time_convergence_w2w3.md   # 可选：把填好的表另存一份
```

---

## Exp #13（2026-09-10 新增）— 汇总 **W2-OPD 100 step** 与 **W3-OPD 50 step** 的逐步诊断

**要求：** 不重训、**不要 PV-OPD**。只统计标准 OPD 两档：W2 按 100 optimizer step、W3 按 50 step。按文末「OPD 逐步诊断」收齐 log、merge、出图，把摘要填回本节。`#12` 管离线 Stage1/KD/RQ；**本实验只管这两条 OPD 曲线**。只报最终 GSM、不交 timeline / 时间，算没做完。

字段口径见文末「OPD 逐步诊断」：`truncation_rate` = response 无 EOS **且** 顶满当时的 `max_length`；跳码 = 相邻 step 整数码变化比例。旧 run 若没有 `opd_step_metrics.jsonl`，状态写 **无 JSONL**，能从 `trainer_state.json` / eval checkpoint 补多少补多少，不要假装有逐步跳码。

W3 的 `#2` 原文是满 epoch Stage 2；**本统计只取前 50 step**（JSONL / `log_history` / eval checkpoint 截到 step≤50）。W2 **固定用 `#10`**（100 step、`2e-6`+wu30 后恒定峰值），不要用 `#5` 默认 `5e-6`，也不要把 `#6` PV-OPD、`#7/#8` 的 50-step 消融写进本表。

| 档 | 统计步数 | tag |
|--|--|--|
| **W2-OPD** | **100** | `#10` `Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold` |
| **W3-OPD** | **50** | `#2` `Qwen3-1.7B-w3g128-opd`（只分析 step 1–50） |

某档没跑到指定步数：写实际 `global_step`，不要拿别的 W2 tag 充数。

### 怎么收

```bash
# W2 100 step = Exp #10
TAG=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
python scripts/merge_opd_timeline.py \
  --metrics log/distill/${TAG}/opd_step_metrics.jsonl \
  --eval-root output/eval/${TAG} \
  --out-dir output/plots/${TAG}
cat log/distill/${TAG}/opd_timing_summary.json

# W3 50 step = Exp #2（merge 后作图/填表只用 step<=50）
TAG=Qwen3-1.7B-w3g128-opd
python scripts/merge_opd_timeline.py \
  --metrics log/distill/${TAG}/opd_step_metrics.jsonl \
  --eval-root output/eval/${TAG} \
  --out-dir output/plots/${TAG}
```

每档至少画（W3 横轴只到 50）：truncation（可叠 eos / response 长）、code-jump + amplification、grad_norm（可叠 lr）、GSM8K / MATH-500。图放 `output/plots/<tag>/`。

### 填表（状态里贴齐）

逐步量取 **起 / 中 / 末**（W2：约 step 1 / 50 / 100；W3：约 1 / 25 / 50）。时间优先 `opd_timing_summary.json`（W3 若跑过 50 step，注明 wall 是全程还是按 50 step 比例估算）。

| 档 | 实际 steps | JSONL | wall / s/step | mean rollout (占比) | trunc 起→末 | jump 起→末 | amp | GSM 起→末 | 判定 |
|--|--|--|--|--|--|--|--|--|--|
| W2-OPD `#10` | 目标 100 | | | | | | | | |
| W3-OPD `#2` | 目标 50 | | | | | | | | |

**判定口（写入状态，可多选）：**

- **策略塌：** truncation / 无 EOS 升高，response 顶满预算  
- **跳码不稳：** jump 高且 GSM 振荡  
- **W2 vs W3：** 同 OPD、不同 bit，100 vs 50 step 下 truncation / jump / rollout 占比差在哪  
- **rollout 太贵：** `rollout_fraction_of_step` 很高  
- **无诊断 log：** 代码早于 JSONL，只能看 eval 点

**依赖：** 训练集群上对应 `log/distill/<tag>` 与 `output/eval/<tag>`。

**状态：** 未填。到有产物的机器上 merge + 出图后，把表和路径写回本节。

产出：

```
output/plots/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/
output/plots/Qwen3-1.7B-w3g128-opd/
output/plots/opd_diagnostics_w2_100_w3_50.md
```

---

## Exp #14（2026-09-10 新增）— Qwen3-**4B** 复现 1.7B 的六档：W3/W2 × Stg1 / KD / ReasoningQAT

**要求：** 把 1.7B 上已经对齐的离线六档原样换到 **Qwen3-4B**。只改模型；**其余超参与对应 1.7B Exp 相同**。不要 OPD，不要 `--train-emb`。产出 stem 必须是 `Qwen3-4B`，**不要覆盖** 1.7B 的 `Qwen3-1.7B-w*g128*`。本实验覆盖公共约定里的「模型是 1.7B」以及 W2 时「不要把 `--wbits` 改成 2」。

| 档 | 对齐的 1.7B Exp | 脚本 | bits | Stage 2 loss | Stage 1 lr | Stage 2 lr / epoch | 产出 |
|--|--|--|--|--|--|--|--|
| **W3-Stg1** | `#1/#9` Stage 1 | 任一带 `--stage 1` | 3 | — | `1e-5` | — | `Qwen3-4B-w3g128` |
| **W3-KD** | `#1` | `run_qwen3_1.7b.sh` | 3 | plain CE + JSD | 同上 | `1e-6` / 1（512 step） | `Qwen3-4B-w3g128` |
| **W3-ReasoningQAT** | `#9` | `run_qwen3_1.7b_reasoningqat.sh` | 3 | teacher-weighted CE + `forward_kl` + cosine | 同上 | `1e-6` / 1 | `Qwen3-4B-w3g128-reasoningqat` |
| **W2-Stg1** | `#4/#11` Stage 1 | `--wbits 2 --stage 1` | 2 | — | `2e-5` | — | `Qwen3-4B-w2g128` |
| **W2-KD** | `#4` | `run_qwen3_1.7b.sh --wbits 2` | 2 | plain CE + JSD | 同上 | `5e-6` / 3（1536 step） | `Qwen3-4B-w2g128` |
| **W2-ReasoningQAT** | `#11` | `run_qwen3_1.7b_reasoningqat.sh --wbits 2` | 2 | 同 `#9` | 同上 | `5e-6` / 3 | `Qwen3-4B-w2g128-reasoningqat` |

锁死与 1.7B 相同：OpenThoughts 32768、effective batch **64**、`max_length=8192`、`top_k=20`、**8 卡** Stage 2（`--gpus 0,1,2,3,4,5,6,7`，accum=8）、Stage 1 单卡。Teacher = 同一份 BF16 Qwen3-4B。不要改 batch / lr / epoch。

**依赖：** 8 卡集群上的 Qwen3-4B 目录（须有 `config.json`）。`MODEL` 按那台机器改，不要照抄本仓库默认的 1.7B 路径。不读 1.7B 的 Stage 1。

**状态：** 未跑。脚本已加 `--exp-name`（默认仍 `Qwen3-1.7B`）。4B **必须** `--exp-name Qwen3-4B`，否则会拒绝以免覆盖 1.7B。Stage 2 若 8192 OOM：该次命令加 `--max-length 4096` 并在状态里注明，不要默默改 batch。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

MODEL=/改成8卡集群上的/Qwen3-4B
test -f "${MODEL}/config.json"

# --- W3 ---
# Stg1（#1 与 #9 共用）
bash scripts/run_qwen3_1.7b.sh --wbits 3 --stage 1 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B

# Stg1 评测（对齐 #3）
CUDA_VISIBLE_DEVICES="${CONVERT_GPU:-0}" python scripts/convert_to_hf_vllm_compatible_model.py \
  --base-id ./output/block_qat/Qwen3-4B-w3g128 \
  --save-dir ./output/vllm/Qwen3-4B-w3g128-blockqat \
  --wbits 3 --group-size 128
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w3g128-blockqat \
  ./output/eval/Qwen3-4B-w3g128-blockqat

# W3-KD（#1）
bash scripts/run_qwen3_1.7b.sh --wbits 3 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B
bash scripts/run_qwen3_1.7b.sh --wbits 3 --stage 3 \
  --model "${MODEL}" --exp-name Qwen3-4B
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w3g128 \
  ./output/eval/Qwen3-4B-w3g128

# W3-ReasoningQAT（#9；复用上面这份 Stage 1，不要重训）
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 3 --stage 3 \
  --model "${MODEL}" --exp-name Qwen3-4B
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w3g128-reasoningqat \
  ./output/eval/Qwen3-4B-w3g128-reasoningqat

# --- W2 ---
bash scripts/run_qwen3_1.7b.sh --wbits 2 --stage 1 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B

CUDA_VISIBLE_DEVICES="${CONVERT_GPU:-0}" python scripts/convert_to_hf_vllm_compatible_model.py \
  --base-id ./output/block_qat/Qwen3-4B-w2g128 \
  --save-dir ./output/vllm/Qwen3-4B-w2g128-blockqat \
  --wbits 2 --group-size 128
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w2g128-blockqat \
  ./output/eval/Qwen3-4B-w2g128-blockqat

bash scripts/run_qwen3_1.7b.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B
bash scripts/run_qwen3_1.7b.sh --wbits 2 --stage 3 \
  --model "${MODEL}" --exp-name Qwen3-4B
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w2g128 \
  ./output/eval/Qwen3-4B-w2g128

bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --model "${MODEL}" --teacher "${MODEL}" --exp-name Qwen3-4B
bash scripts/run_qwen3_1.7b_reasoningqat.sh --wbits 2 --stage 3 \
  --model "${MODEL}" --exp-name Qwen3-4B
bash scripts/eval_paper_benchmarks.sh \
  ./output/vllm/Qwen3-4B-w2g128-reasoningqat \
  ./output/eval/Qwen3-4B-w2g128-reasoningqat
```

建议顺序：W3-Stg1 → 评 Stg1 → W3-KD 与 W3-RQ 可并行（同 Stage 1）→ 同样做 W2。训完按 `#12` 口径记墙钟 / loss。

产出：

```
output/block_qat/Qwen3-4B-w3g128
output/distill/Qwen3-4B-w3g128
output/distill/Qwen3-4B-w3g128-reasoningqat
output/block_qat/Qwen3-4B-w2g128
output/distill/Qwen3-4B-w2g128
output/distill/Qwen3-4B-w2g128-reasoningqat
output/eval/Qwen3-4B-w3g128-blockqat
output/eval/Qwen3-4B-w3g128
output/eval/Qwen3-4B-w3g128-reasoningqat
output/eval/Qwen3-4B-w2g128-blockqat
output/eval/Qwen3-4B-w2g128
output/eval/Qwen3-4B-w2g128-reasoningqat
```

和 1.7B 同名后缀的 eval 对比。不要覆盖 `Qwen3-1.7B-*`。

---

## Exp #15（2026-09-10 新增）— 从 `#10` 的 **step 30** 起 hold `2e-6` 把后面跑完

**要求：** `#10` 前 30 step warmup（`2e-7→2e-6`）是对的，从 31 起本应 **钉住 `2e-6`**，实际在 decay。本实验 **不要重训 1–30，也不要从 100 接着训**。在 **跑 `#10` 的那台 8 卡集群**上，用修好 hold 的代码，从 `#10` 的 Trainer **`checkpoint-30`** 续跑到 100（原来设计的 70 步 stable）。不要 `--train-emb`，不要重训 Stage 1。`--run-suffix from30hold` 写到新目录，**不要覆盖** `#10` 已经 decay 的 31–100。本实验覆盖公共约定里的「不要把 `--wbits` 改成 2」。

| | `#10` | **本实验 `#15`** |
|--|--|--|
| step 1–30 | warmup 到 `2e-6`（保留） | **不重跑，加载 8 卡机上的 ckpt-30** |
| step 31–100 | 实际 linear → 0 | **hold `2e-6` 跑完这 70 步** |
| 起点 | Stage 1 | **`#10` 的 step 30** |
| 产出 | `…-schhold` | **`…-schhold-from30hold`** |

其余与 `#10` 相同：W2、CE=0、sampled reverse KL、8 卡、batch 64、`max_length=8192`、每 5 step 存 ckpt。`--max-steps 100`（从 global_step=30 训到 100）。不要 `--resume`（会捡 step 100）。必须 `--resume-from log/distill/<tag10>/checkpoint-30`。

**开训前闸门：**

1. 在 8 卡集群拉到含 `scheduler_type_name` 的代码；rank0 要出现 `hold_after_warmup=True`，否则停。
2. `test -d log/distill/${TAG10}/checkpoint-30` 必须过（Trainer 全量，不是 `output/distill` 评测快照）。没有就停下来找，不要改成 `--init-from`。
3. 续训后 step 35 / 50 / 75 / 100 的 `learning_rate` 都必须 ≈ `2e-6`。再掉下去作废。

**依赖：** 8 卡集群上的 `#4` Stage 1，以及 `#10` 的 `log/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-30`。

**状态（2026-09-14）：** 已从 `#10 checkpoint-30` 续训到 **step 62/100**；
hold 修复生效，step 31–62 的 `learning_rate=2e-6`。但三次重试都在即将进入
step 63 时卡在同一个 DDP `ALLREDUCE`，最终由 2 h watchdog 退出。卡死前
`truncation_rate=0`、loss/grad_norm 有限，无 NaN，不按梯度爆炸处理。

**已定位根因：** selected OpenThoughts 中存在 tokenized prompt length
`>=8192` 的样本；在 `max_length=8192`、`min_rollout_tokens=1` 下没有任何
response 生成空间。当前 `PolicyGKDTrainer.training_step` 只让拿到该样本的
本地 rank 提前 `return loss`，该 rank 没有执行 backward，其他 rank 继续进入
梯度 `ALLREDUCE`，因此确定性死锁。日志里报 timeout 的 rank 不一定是拿到
超长 prompt 的 rank。

**修复与续训要求（覆盖下面原始“从 checkpoint-30 开跑”的命令）：**

1. 在创建 Trainer / DistributedSampler **之前**，用与 collator 相同的 chat
   template 和 tokenizer 计算 prompt-only 长度，统一过滤
   `prompt_length >= 8192`（一般式：`prompt_length > max_length - min_rollout_tokens`）。
   不要按原始字符串长度过滤，也不要把 gold completion 算进 prompt 长度。
2. 所有 rank 必须看到同一份过滤后的 dataset。启动日志写出过滤前后样本数、
   删除数和剩余最大 prompt length；剩余最大值必须 `<=8191`。不能再依赖
   rank-local early return 绕过超长样本。
3. 停止原样重试；保留已有 step 35/50 评测。从本实验自己的 Trainer
   `checkpoint-60` 继续到 100，保留 Adam、RNG 和 hold scheduler。不要回到
   `#10 checkpoint-30`，不要从评测快照 `output/distill/.../checkpoint-60`
   热启动。
4. 过滤会使 step 60 之后的数据流相对原 run 少掉超长样本；在最终状态中记录
   `filtered_count`。这比改 seed / `dataset_start_index` 更可解释。
5. 恢复后先确认越过 step 63，且 step 65/75/100 的 LR 都约为 `2e-6`；随后补评
   75/100 并执行 timeline merge。若仍在 step 63 卡住，停止重试并收集各 rank
   在 generate/backward 前后的日志。

续训命令（代码含上述全局预过滤后执行）：

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

TAG10=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
TAG15=${TAG10}-from30hold
test -d log/distill/${TAG15}/checkpoint-60

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 100 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --resume-from log/distill/${TAG15}/checkpoint-60 \
  --run-suffix from30hold

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/${TAG15} \
  --wbits 2 --eval-gpu 0 \
  --steps 75,100 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics log/distill/${TAG15}/opd_step_metrics.jsonl \
  --eval-root output/eval/${TAG15} \
  --out-dir output/plots/${TAG15}
```

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

TAG10=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
TAG15=${TAG10}-from30hold
test -f output/block_qat/Qwen3-1.7B-w2g128/config.json
test -d log/distill/${TAG10}/checkpoint-30

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 100 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --resume-from log/distill/${TAG10}/checkpoint-30 \
  --run-suffix from30hold

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir ./output/distill/${TAG15} \
  --wbits 2 --eval-gpu 0 \
  --steps 35,50,75,100 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics log/distill/${TAG15}/opd_step_metrics.jsonl \
  --eval-root output/eval/${TAG15} \
  --out-dir output/plots/${TAG15}
```

横幅：`Resuming training from checkpoint` 指向 **checkpoint-30**（不是 100）、`max_steps=100`、`hold_after_warmup=True`。`learning_rate` 从 31 到 100 应是 `2e-6` 平线。

产出：

```
output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
output/eval/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
log/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
output/plots/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
```

对比同一 step 的 `#10`（decay）vs 本实验（hold）。不要覆盖 `#10` 的 `…-schhold`。

---

## Exp #16（2026-09-13 新增）— W2 ReasoningQAT 最终权重再训练 **128 steps**

**要求：** 从 `#11` 的最终 W2 ReasoningQAT 权重热启动，额外训练 128 个 optimizer steps，判断离线 ReasoningQAT 是否仍有增益。loss 保持 `0.2 * teacher-weighted CE + 1.0 * forward_kl`、`top_k=20`；尾段用 `lr=1e-6`、无 warmup、cosine 衰减到 0。每 32 step 保存一次，所有 checkpoint 和最终模型**只评 GSM8K 与 MATH-500**。不要覆盖 `#11`。

这不是严格的 Trainer resume：`#11` 默认只保存最终量化权重，没有 Adam/scheduler checkpoint。因此本实验会保留模型权重，但重置 Adam 和学习率调度器；这里的 `max_steps=128` 表示**额外**训练 128 steps。

**依赖：** `output/distill/Qwen3-1.7B-w2g128-reasoningqat/config.json`。

**状态：** 未跑。在 8 卡机上跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

BASE=Qwen3-1.7B-w2g128-reasoningqat
TAG=${BASE}-cont128
test -f output/distill/${BASE}/config.json

mkdir -p output/distill/${TAG} log/distill/${TAG}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file configs/accelerate_config_multigpu.yaml \
  --num_processes 8 --gpu_ids all \
  main_e2e_distill.py \
  --model output/distill/${BASE} \
  --teacher_model "${MODEL_PATH}" \
  --wbits 2 --group_size 128 \
  --epochs 3 --max_steps 128 \
  --learning_rate 1e-6 --lr_scheduler_type cosine --warmup_steps 0 \
  --gkd_beta 1.0 --kl_weight 1.0 --cross_entropy_weight 0.2 \
  --use_teacher_weight --kd_loss_type forward_kl --top_k 20 \
  --dataset_type openthoughts --dataset_size 32768 --max_length 8192 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --save_steps 32 --save_total_limit 4 \
  --save_quant_dir output/distill/${TAG} \
  --output_dir log/distill/${TAG}

touch output/distill/${TAG}/.train_done

# 评 32/64/96/128 step 和最终权重；只跑 GSM8K、MATH-500，保留快照。
KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir output/distill/${TAG} \
  --wbits 2 --eval-gpu 0 \
  --steps 32,64,96,128
```

开跑前应确认：`model=...-reasoningqat`、`max_steps=128`、`save_steps=32`、`learning_rate=1e-6`、`warmup_steps=0`。主要比较 `#11` final 与本实验 step 32/64/96/128 的 GSM8K、MATH-500；若提升只在中间出现，以最佳 checkpoint 为准，不要只报 final。

产出：

```
output/distill/Qwen3-1.7B-w2g128-reasoningqat-cont128
log/distill/Qwen3-1.7B-w2g128-reasoningqat-cont128
output/eval/Qwen3-1.7B-w2g128-reasoningqat-cont128
```

---

## Exp #17（2026-09-11 新增）— BF16 Qwen3-1.7B **thinking on** 的 GSM8K 上界

**要求：** 评 **未量化、未训练** 的 BF16 `Qwen3-1.7B` 原模，**必须开 thinking**。用来当 `#10` / `#15` 以及后续量化实验的 teacher 上界。不要训练，不要 Stage 1/2，不要 convert，不要 `--train-emb`。只跑评测。本实验覆盖公共约定里「评的是量化产出」：这里评的是 `$MODEL_PATH` 原权重。

协议与公共约定相同：`eval_paper_benchmarks.sh` + vLLM + evalscope，`T=0.6`，`top_k=20`，`max_tokens=8192`。**必须** `ENABLE_THINKING=1`。**不要**用 `scripts/eval_gsm8k_aime120.py`（那条是 thinking off + greedy + 2048）。

| | OPT-QAT S0（已有，**不是本实验**） | **本实验 `#17`** |
|--|--|--|
| 模型 | 同一份 BF16 `Qwen3-1.7B` | **相同** |
| thinking | **关** | **开** |
| 采样 | greedy，`max_new_tokens=2048` | **T=0.6 / top_k=20 / 8192**（公共约定） |
| 任务 | GSM8K 75.7%（999/1319） | **GSM8K 全量 1319**（必做） |
| 产出 | `OPT-QAT/outputs/eval_qwen3_1p7b_s0_bf16` | **`output/eval/Qwen3-1.7B-bf16-thinking`** |

主交付是 GSM8K。有余力再把 `EVAL_DATASETS` 改成 `gsm8k math_500`，不要一上来跑满 8 项。单卡即可，不要占满 8 卡。

**开跑后闸门：** 前几条 generation 必须出现 `<think>`。没有就停，不要把 thinking-off 的分数写进状态。

**依赖：** `$MODEL_PATH/config.json`（BF16 原模）。无训练产物依赖。

**状态：** 未跑。这台 4 卡机（`ajv59d3mbdufs-0`）经常被占满，**不要抢别人的卡**。到有 **1 张空闲 GPU** 的机器上跑（8 卡集群空 1 张即可）。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f "${MODEL_PATH}/config.json"

# 只占 1 张空闲卡。把 0 改成 nvidia-smi 里空的那张。
export CUDA_VISIBLE_DEVICES=0

ENABLE_THINKING=1 \
EVAL_DATASETS=gsm8k \
MAX_TOKENS=8192 \
EVAL_BATCH_SIZE=16 \
  bash scripts/eval_paper_benchmarks.sh \
  "${MODEL_PATH}" \
  ./output/eval/Qwen3-1.7B-bf16-thinking
```

横幅应有 `enable_thinking=1`、`datasets: gsm8k`、`max_tokens=8192`。产出不要覆盖任何 `Qwen3-1.7B-w*` 的 eval。

产出：

```
output/eval/Qwen3-1.7B-bf16-thinking
```

对比：OPT-QAT S0 的 thinking-off 75.7%；以及 `#10` 量化后的 GSM8K 曲线。状态里写 GSM8K 分 + 评测墙钟 + 是否确认过 `<think>`。没有 Stage 2，不必交 OPD timeline。

---

## Exp #18（2026-09-14 新增）— W2 Stability-Gated Mixed OPD 试跑

**目标：** 从 `#4` 的 W2 Stage1 权重出发，用一组最小的
固定混合对照检验「offline trajectory anchor + code-jump gate」是否值得继续。
这是 screening，不作为最终论文结果；只有通过下面 go/no-go 条件才扩到 100
step 和多 seed。

两条分支除 occupancy 调度外完全一致：OpenThoughts 32768、effective batch 64、
`max_length=8192`、CE=0、peak LR `2e-6`、前 30 step 从 `2e-7` warmup，之后
hold。offline step 在 gold/teacher trajectory 上算 teacher top-20 forward KL；
online step 使用 student rollout + 当前 sampled reverse-KL。不要 `--train-emb`。

| 分支 | 初始 online ratio | gate | 回答的问题 |
|---|---:|---|---|
| A fixed-50 | 0.50 | 关闭（`interval=0`） | 混合 occupancy 本身是否有效 |
| B gated | 0.25 | EMA；soft=5%，hard=8% | 动态门控是否优于固定混合 |

B 每 5 optimizer step 决策一次，step 30 前只记录、不改变比例；连续两个决策
窗口的 jump EMA `<5%` 后把 online ratio 加 0.25；raw jump 或 jump EMA
`>=8%` 时立即减 0.25，比例下限 0.25、上限 1.0。门控只改 occupancy，
**不同时改 LR**。
同一 optimizer step 的所有 gradient-accumulation microbatch 使用同一种
occupancy；rank 0 计算 jump 后把下一步 ratio 同步给全部 DDP rank。

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`（`#4` Stage1）。
两条分支都从同一份 Stage1 权重启动；不要使用 `--init-from` 或 `--resume`。
启动前代码单测和 shell 检查必须通过。

**状态：** 未跑。按两阶段顺序筛选，先只跑 B 到 40 step；B 没信号就停止，
不要支付 A 的训练成本。B 通过 screening 后才跑 A。不要并行争抢资源。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

STAGE1=output/block_qat/Qwen3-1.7B-w2g128
test -f "${STAGE1}/config.json"

python tests/test_opd_stage2.py
bash -n scripts/run_qwen3_1.7b_opd.sh

# Phase 1 / B：先只跑 5%/8% code-jump EMA 门控筛选。
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mixed-opd --online-ratio 0.25 \
  --gate-soft 0.05 --gate-hard 0.08 --gate-ratio-step 0.25 \
  --gate-interval 5 --gate-patience 2 --gate-start-step 30 \
  --gate-ema-beta 0.9 \
  --max-steps 40 --save-steps 20 \
  --lr 2e-6 --warmup-steps 30 --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --top-k 20 --run-suffix stage1-gated-screen40

# Phase 2 / A：只有 B 通过 screening 才运行这个固定 50% 对照。
bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mixed-opd --online-ratio 0.50 --gate-interval 0 \
  --max-steps 40 --save-steps 20 \
  --lr 2e-6 --warmup-steps 30 --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --top-k 20 --run-suffix stage1-fixed50-screen40
```

训练启动后先检查横幅和日志：

1. 两条分支都必须显示 `student=.../block_qat/Qwen3-1.7B-w2g128`、CE=0、offline=top20-FKL、
   online=sampled-RKL、hold=True。
2. prompt filter 必须打印 before/after/filtered/max；`max_after<=8191`，否则停。
3. A 的 `opd/online_ratio` 始终为 0.50；B 在 step 30 前始终为 0.25。
4. JSONL 必须含 `opd/online_batch`、`opd/online_ratio`、
   `opd/code_jump_ema`、`opd/gate_action`，且所有 rank 不得走不同 occupancy。

Phase 1 只评 B 的 step 40，并补齐 Vanilla step 40；不要先评中间点：

```bash
FIXED=Qwen3-1.7B-w2g128-mixedopd-lr2e-6-wu30-ws2e-7-schhold-stage1-fixed50-screen40
GATED=Qwen3-1.7B-w2g128-mixedopd-lr2e-6-wu30-ws2e-7-schhold-stage1-gated-screen40
VANILLA=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold

# #15 checkpoint-40 是相同 Stage1/LR schedule 的 online-only 对照。
KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${VANILLA}" \
  --wbits 2 --eval-gpu 0 --steps 40 --skip-final

KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${GATED}" \
  --wbits 2 --eval-gpu 0 --steps 40 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${GATED}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${GATED}" \
  --out-dir "output/plots/${GATED}"

# B 通过下面 screening 后，再用同一命令评 FIXED 的 step 40。
```

**Phase 1 go/no-go：** B 在 step 40 前必须至少改变一次 ratio、jump P95 `<8%`、
peak `<10%`、无 NaN/分数崩塌；并且相对同 step Vanilla 满足：两任务均值不差
超过 0.5 pt，或 jump P95 至少降低 20%。不满足就停止，不跑 A。

**Phase 2 go/no-go：** B 通过后才跑 A。相对 A，B 的两任务均值高 1.0 pt，
或者 jump P95 降低 20% 且均值不差超过 0.5 pt，才把 gate 视为有效。另报累计
online steps 和 rollout wall time，并按相同累计 online steps 对齐比较。刚过线的
checkpoint 重复评一次，避免把约 1 pt 的采样噪声当成提升。

- B 通过两阶段筛选：重新从 Stage1 跑正式 60/100 step，再补 3 seeds；随后做 W3。
- A、B 都提升但相近：保留更简单的 fixed mixture，动态 gate 暂不作为贡献。
- A 提升、B 变差：阈值/滞后有问题，只扫 `4/6`、`5/8`、`6/10` 三档。
- A、B 都不提升：停止调 gate，先做 loss-matched Offline-FKL vs Online-FKL，
  判断瓶颈是 occupancy 还是 sampled-RKL objective。

产出：

```
output/distill/<FIXED|GATED>
output/eval/<FIXED|GATED>
log/distill/<FIXED|GATED>/opd_step_metrics.jsonl
log/distill/<FIXED|GATED>/opd_timing_summary.json
output/plots/<FIXED|GATED>/opd_timeline.csv
```

---

## Exp #19（2026-09-14 新增）— 对比 W2-OPD 的 decay70 与 stable55 训练统计

**目标：** 不重训，按每 5 optimizer steps 汇总已有的两条 W2-OPD 轨迹，
判断 stable 相比 decay 是否真的降低了梯度/跳码波动，并对应到同 checkpoint
的 GSM8K、MATH-500。主比较使用共同终点 step 85；不能拿 stable 的 step 85
直接和 decay 的 step 100 下结论。

| 简称 | 数据组成 | 实际 schedule |
|---|---|---|
| decay70 | `#10` step 1–100 | warmup 1–30 + decay 31–100 |
| stable55 | `#10` step 1–30 + `#15` step 31–85 | 同一 warmup + hold `2e-6` 55 step |

`#15` 若已经跑过 step 85，统计时仍截断到 85。JSONL 中 resume/retry 造成的重复
step 按字段保留最后一次记录；最终 `train_runtime` 等稀疏日志不能覆盖该 step 已有
训练指标。

**统计口径：**

- 完整 schedule：decay70 统计 step 1–100；stable55 统计拼接后的 step 1–85。
- tail 主分析：decay70 统计 step 31–100；stable55 统计 step 31–85。
- **每 5 steps 主表：** checkpoint 5 对应 step 1–5，checkpoint 35 对应
  step 31–35，以此类推；decay 输出到 100，stable 输出到 85。不能只抽取
  checkpoint 当步的瞬时训练值。
- LR、loss、response length、EOS、truncation 报 step 宏平均；grad norm、code
  jump、code-jump amplification 报均值 / P95 / 最大值。P95 用线性插值。
- `code_jump_rate`、`eos_rate`、`truncation_rate` 以百分数报告；amplification
  为无量纲比值。每项同时打印有效 step 数，数量不足时不填结果。
- 评测协议沿用原实验：thinking on、`T=0.6`、`top_k=20`、
  `max_tokens=8192`。decay 每 5 steps 评到 100；stable 的 1–30 与 decay
  共用结果，独立分支从 35 每 5 steps 评到 85。

**依赖：** `#10` 的完整 JSONL / checkpoints，以及 `#15` 至少训练并保存到
step 85。没有 step 85 就先补完 `#15`，不要用插值后的 benchmark 分数。

**状态：** 待统计。只做日志聚合和缺失 checkpoint 评测，不启动 Stage 2。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

DECAY=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold
STABLE=${DECAY}-from30hold

test -f "log/distill/${DECAY}/opd_step_metrics.jsonl"
test -f "log/distill/${STABLE}/opd_step_metrics.jsonl"
test -d "output/distill/${DECAY}/checkpoint-85"
test -d "output/distill/${STABLE}/checkpoint-85"

# 补齐每 5-step 评测；已有结果会复用。step 5–30 是两条分支的共享 warmup。
KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${DECAY}" --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
  --skip-final

KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${STABLE}" --wbits 2 --eval-gpu 0 \
  --steps 35,40,45,50,55,60,65,70,75,80,85 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${DECAY}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${DECAY}" \
  --out-dir "output/plots/${DECAY}"

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${STABLE}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${STABLE}" \
  --out-dir "output/plots/${STABLE}"

# 打印全程摘要、每 5-step 统计和 checkpoint 评测表；只使用 Python 标准库。
python - <<'PY'
import json
import math
from pathlib import Path
from quantize.opd_metrics import load_eval_scores_for_step

decay = "Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold"
stable = decay + "-from30hold"

keys = {
    "lr": "learning_rate",
    "loss": "loss",
    "grad": "grad_norm",
    "jump": "opd/code_jump_rate",
    "amp": "opd/code_jump_amplification",
    "response": "opd/mean_response_len",
    "eos": "opd/eos_rate",
    "trunc": "opd/truncation_rate",
}

def load(tag):
    # Later records overwrite only fields they actually contain. This keeps the
    # last retry while preventing a final sparse train summary from erasing data.
    rows = {}
    path = Path("log/distill") / tag / "opd_step_metrics.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "step" in row:
            rows.setdefault(int(row["step"]), {}).update(row)
    return rows

def pct95(values):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = 0.95 * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def values(rows, key, lo, hi):
    return [float(rows[s][key]) for s in range(lo, hi + 1)
            if s in rows and key in rows[s]]

def fmt(x, percent=False):
    return f"{100*x:.4f}%" if percent else f"{x:.6g}"

def report(name, rows, lo, hi):
    xs = {name: values(rows, key, lo, hi) for name, key in keys.items()}
    required = hi - lo + 1
    counts = ", ".join(f"{k}={len(v)}/{required}" for k, v in xs.items())
    if any(len(v) != required for v in xs.values()):
        raise SystemExit(f"{name} step {lo}-{hi}: missing metrics: {counts}")
    mean = lambda v: sum(v) / len(v)
    return {
        "run": name, "steps": f"{lo}-{hi}", "ckpt": hi, "n": required,
        "lr_mean": fmt(mean(xs["lr"])), "loss_mean": fmt(mean(xs["loss"])),
        "grad_mean": fmt(mean(xs["grad"])),
        "grad_p95": fmt(pct95(xs["grad"])), "grad_max": fmt(max(xs["grad"])),
        "jump_mean": fmt(mean(xs["jump"]), True),
        "jump_p95": fmt(pct95(xs["jump"]), True),
        "jump_max": fmt(max(xs["jump"]), True),
        "amp_mean": fmt(mean(xs["amp"])),
        "amp_p95": fmt(pct95(xs["amp"])), "amp_max": fmt(max(xs["amp"])),
        "response_mean": fmt(mean(xs["response"])),
        "eos_mean": fmt(mean(xs["eos"]), True),
        "trunc_mean": fmt(mean(xs["trunc"]), True),
    }

d, s = load(decay), load(stable)
stable_full = {k: v for k, v in d.items() if 1 <= k <= 30}
stable_full.update({k: v for k, v in s.items() if 31 <= k <= 85})

summary_rows = []
for scope, reports in [
    ("完整 schedule", [report("decay70", d, 1, 100),
                       report("stable55", stable_full, 1, 85)]),
    ("tail（主分析）", [report("decay70", d, 31, 100),
                       report("stable55", s, 31, 85)]),
]:
    summary_rows.extend({"scope": scope, **row} for row in reports)

window_rows = []
for name, data, stop in [("decay70", d, 100),
                         ("stable55", stable_full, 85)]:
    window_rows.extend(report(name, data, end - 4, end)
                       for end in range(5, stop + 1, 5))

def table(title, table_rows, columns):
    print(f"\n### {title}\n")
    print("| " + " | ".join(label for _, label in columns) + " |")
    print("|" + "---|" * len(columns))
    for row in table_rows:
        print("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")

summary_base = [("scope", "统计区间"), ("run", "run"),
                ("steps", "steps"), ("n", "n")]
window_base = [("ckpt", "checkpoint"), ("run", "run"),
               ("steps", "统计 steps")]
optimization = [
    ("lr_mean", "LR mean"), ("loss_mean", "loss mean"),
    ("grad_mean", "grad mean"), ("grad_p95", "grad P95"),
    ("grad_max", "grad max"),
]
quantization = [
    ("jump_mean", "jump mean"), ("jump_p95", "jump P95"),
    ("jump_max", "jump max"), ("amp_mean", "amp mean"),
    ("amp_p95", "amp P95"), ("amp_max", "amp max"),
]
rollout = [
    ("response_mean", "response length mean"),
    ("eos_mean", "EOS mean"), ("trunc_mean", "truncation mean"),
]
table("表 1：优化状态摘要", summary_rows, summary_base + optimization)
table("表 2：每 5 steps 优化状态", window_rows, window_base + optimization)
table("表 3：量化稳定性摘要", summary_rows, summary_base + quantization)
table("表 4：每 5 steps 量化稳定性", window_rows, window_base + quantization)
table("表 5：Rollout 状态摘要", summary_rows, summary_base + rollout)
table("表 6：每 5 steps Rollout 状态", window_rows, window_base + rollout)

def eval_score(tag, step, dataset):
    scores = load_eval_scores_for_step(Path("output/eval") / tag, step)
    candidates = [(k, v) for k, v in scores.items()
                  if dataset in k.lower()]
    if not candidates:
        return "—"
    # Prefer the shortest dataset-specific accuracy/score key over nested copies.
    candidates.sort(key=lambda kv: (
        not kv[0].lower().endswith(("accuracy", "score", "exact_match")),
        len(kv[0]),
    ))
    value = float(candidates[0][1])
    value = 100 * value if abs(value) <= 1 else value
    return f"{value:.2f}"

print("\n### 表 7：每 5 steps 的 GSM8K / MATH-500\n")
print("| step | decay GSM8K | decay MATH-500 | stable GSM8K | stable MATH-500 |")
print("|---:|---:|---:|---:|---:|")
for step in range(5, 101, 5):
    stable_tag = decay if step <= 30 else stable
    stable_scores = ([eval_score(stable_tag, step, ds)
                      for ds in ("gsm8k", "math_500")]
                     if step <= 85 else ["—", "—"])
    row = [step, eval_score(decay, step, "gsm8k"),
           eval_score(decay, step, "math_500"), *stable_scores]
    print("| " + " | ".join(map(str, row)) + " |")
PY
```

把脚本输出的七张表填回本节。摘要表用于总体结论；每 5-step 表用于检查
“grad/jump 尖峰 → rollout 改变 → 后续 benchmark 改变”的时间顺序。表 7 格式：

| step | decay70 GSM8K | decay70 MATH-500 | stable55 GSM8K | stable55 MATH-500 |
|---:|---:|---:|---:|---:|
| 5–30（每 5 step） | 待填 | 待填 | 同左 | 同左 |
| 35–80（每 5 step） | 待填 | 待填 | 待填 | 待填 |
| **85（主比较）** | **待填** | **待填** | **待填** | **待填** |
| 90–100（每 5 step，仅 decay） | 待填 | 待填 | — | — |

**结论门槛：** 只有 stable 在共同 step 85 的 GSM8K/MATH-500 不差、且 tail
的 grad norm 或 code jump 的 P95/最大值明显更低，才支持“stable 提高训练稳定性”。
若仅 LR/loss 更平滑而 benchmark 不升，只能说优化轨迹更平滑，不能说效果更好。

产出：

```
output/plots/${DECAY}/opd_timeline.{csv,jsonl}
output/plots/${STABLE}/opd_timeline.{csv,jsonl}
```

---

## Exp #20（2026-09-15 新增）— W2-OPD stable step 70–85 完整 benchmark

**目标：** 不重训。对 `#15/#19` 的 W2-OPD stable55 分支中
checkpoint 70/75/80/85 跑完整 8 项 benchmark，判断 GSM8K/MATH-500 的最佳
窗口是否同时改善通用知识、代码、指令遵循和困难推理，避免只凭两个任务选择
checkpoint。

完整套件固定为：GSM8K、AIME24、AIME25、MATH-500、LiveCodeBench、
MMLU-Redux、GPQA-Diamond、IFEval。评测配置与公共约定一致：thinking on、
`T=0.6`、`top_k=20`、`max_tokens=8192`。四个 checkpoint 必须使用完全相同的
配置；不要设置 `LIMIT`，也不要设置 `EVAL_DATASETS` 子集。

**依赖：** 以下四个评测快照均存在且带 `.ready`：

```
output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-{70,75,80,85}
```

**状态：** 未跑。只做 checkpoint 转换和评测，不启动 Stage 1/2。已有
GSM8K/MATH-500 的 `.eval_done` 是子集评测标记，必须清掉；完整结果写入新的
`-fullbench70to85` tag，不覆盖 Exp #19 的双任务结果。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

STABLE=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
FULL_TAG=${STABLE}-fullbench70to85

for STEP in 70 75 80 85; do
  test -f "output/distill/${STABLE}/checkpoint-${STEP}/config.json"
  test -f "output/distill/${STABLE}/checkpoint-${STEP}/.ready"
done

# watcher 的完成标记不区分评测子集；清除这四个旧标记后才能补跑完整套件。
rm -f output/distill/${STABLE}/checkpoint-{70,75,80,85}/.eval_done
unset EVAL_DATASETS LIMIT

KEEP_CHECKPOINTS=1 \
ENABLE_THINKING=1 \
MAX_TOKENS=8192 \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${STABLE}" \
  --tag "${FULL_TAG}" \
  --wbits 2 --eval-gpu 0 \
  --steps 70,75,80,85 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${STABLE}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${FULL_TAG}" \
  --out-dir "output/plots/${FULL_TAG}"

# 打印完整结果表；paper avg 只平均论文主表五任务：
# MATH-500 / LCB / MMLU-R / GPQA-D / IFEval。
python - <<'PY'
from pathlib import Path
from quantize.opd_metrics import load_eval_scores_for_step

tag = ("Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-"
       "from30hold-fullbench70to85")
datasets = [
    ("gsm8k", "GSM8K"), ("aime24", "AIME24"),
    ("aime25", "AIME25"), ("math_500", "MATH-500"),
    ("live_code_bench", "LCB"), ("mmlu_redux", "MMLU-R"),
    ("gpqa_diamond", "GPQA-D"), ("ifeval", "IFEval"),
]
paper_tasks = {"math_500", "live_code_bench", "mmlu_redux",
               "gpqa_diamond", "ifeval"}

def score(scores, dataset):
    candidates = [(key, value) for key, value in scores.items()
                  if dataset in key.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        not item[0].lower().endswith(
            ("accuracy", "score", "exact_match", "pass@1", "pass_at_1")
        ),
        len(item[0]),
    ))
    value = float(candidates[0][1])
    return 100 * value if abs(value) <= 1 else value

header = ["ckpt", *[label for _, label in datasets], "paper-5 avg"]
print("| " + " | ".join(header) + " |")
print("|" + "---:|" * len(header))
for step in (70, 75, 80, 85):
    scores = load_eval_scores_for_step(Path("output/eval") / tag, step)
    values = {name: score(scores, name) for name, _ in datasets}
    main = [values[name] for name in paper_tasks]
    avg = sum(main) / len(main) if all(v is not None for v in main) else None
    row = [str(step), *[("—" if values[name] is None else f"{values[name]:.2f}")
                        for name, _ in datasets],
           "—" if avg is None else f"{avg:.2f}"]
    print("| " + " | ".join(row) + " |")
PY
```

把输出填回本节：

| ckpt | GSM8K | AIME24 | AIME25 | MATH-500 | LCB | MMLU-R | GPQA-D | IFEval | paper-5 avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 75 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 80 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 85 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

**选择规则：** 主 checkpoint 按 `paper-5 avg` 选择，不按 GSM8K/MATH-500
两任务均值选择。四个点全部报告；若差异小于约 1 pt，至少重复评测候选最佳点，
不要把采样噪声写成确定提升。

产出：

```
output/eval/${FULL_TAG}/checkpoint-{70,75,80,85}
output/plots/${FULL_TAG}/opd_timeline.{csv,jsonl}
log/eval/${FULL_TAG}/watcher.log
```

---

## Exp #21（2026-09-17 新增）— W2-OPD stable 从 step 85 续训 65 step

**目标：** 从 `#15` 的 stable55 Trainer `checkpoint-85` 真续训到 global
step 150，即额外训练 65 个 optimizer steps。检验 stable 分支在 80–85 附近达到
平台后，继续保持 `lr=2e-6` 是继续恢复、震荡，还是过拟合/退化。本实验不改
loss、数据、batch、量化配置或 LR，只增加训练长度。

**固定配置：** W2A16、sampled reverse-KL OPD、CE=0、effective batch 64、
`max_length=8192`、8 GPU；step 85–150 保持 `lr=2e-6`。必须加载 Trainer
checkpoint，保留模型、Adam、scheduler 和 RNG；不能用
`output/distill/.../checkpoint-85` 权重热启动。

**依赖：**

```
log/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-85
```

**状态：** 未跑。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

SRC=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold
RUN21=Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-stable-cont85to150
test -f "log/distill/${SRC}/checkpoint-85/trainer_state.json"

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 150 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --resume-from "log/distill/${SRC}/checkpoint-85" \
  --run-suffix stable-cont85to150

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${RUN21}" \
  --wbits 2 --eval-gpu 0 \
  --steps 90,95,100,105,110,115,120,125,130,135,140,145,150 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${RUN21}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${RUN21}" \
  --out-dir "output/plots/${RUN21}"
```

**开训闸门：** 第一条新日志必须是 global step 86 附近，LR 始终约
`2e-6`；若从 step 0 重启、Adam 被重置或 LR 重新 warmup，立即停止。本实验
只用 GSM8K/MATH-500 筛选趋势；选出最佳点后再补完整 8 项 benchmark。

**必报：** 每 5 step 的 GSM8K/MATH-500，以及 LR、loss、grad norm、code
jump、response length、EOS/truncation；同时报告 step 85→150 的总墙钟。主对照
是 `#20` 的 stable checkpoint 80/85。

产出：

```
log/distill/${RUN21}/checkpoint-{90,95,...,150}
output/distill/${RUN21}/checkpoint-{90,95,...,150}
output/eval/${RUN21}/checkpoint-{90,95,...,150}
output/plots/${RUN21}/opd_timeline.{csv,jsonl}
```

---

## Exp #22（2026-09-17 新增）— W2 ShortOPD-inspired 自适应 rollout，Stage 1 起训 100 step

**目标：** 从 `#4` 的 W2 Stage 1 权重重新训练 100 step，把
[ShortOPD](https://arxiv.org/abs/2607.13124) 的“重复检测 + 截断感知 +
short-to-long rollout budget”移植到当前量化 OPD，检验 W2 的早期长 rollout
是否在重复后缀上浪费计算并产生低价值更新。

这不是 ShortOPD 论文的严格复现。论文使用 top-100+tail generalized JSD 和
teacher-loss 边界细化；本实验为了和 `#15` 做单变量比较，继续使用当前的
**sampled reverse-KL OPD**，只改变 rollout horizon。检测到的重复 token 仍参与
OPD loss，检测器只控制下一 step 的生成预算。

**与 `#15` 对齐的配置：** W2A16、同一 Stage 1、teacher、OpenThoughts 数据、
effective batch 64、8 GPU、CE=0、`T=0.6`、总长上限 8192；LR 为前 30 step
`2e-7→2e-6`，之后 hold `2e-6`，总计 100 step。唯一实验变量是 response
budget (H_t)。

**ShortOPD controller：**

| 项 | 配置 |
|--|--|
| 初始/最大 budget | `8192` response tokens；实际每批为 `min(H_t, 8192 - padded_prompt_len)` |
| 最小 budget | `1024` |
| terminal loop | 最后 512 token，period 1–10，末端 32 token agreement ≥0.9 |
| severe loop | tail ≥128 token，或占 suffix ≥30%；至少 3 cycles、tail ≥64 |
| repetition gate | `rho_low=0.20`, `rho_high=0.45` |
| clean truncation gate | `tau=0.10` |
| effective-length margin / growth | `1.15 / 1.25` |
| statistic EMA / budget EMA | `0.7 / 0.7` |
| budget rounding | 16-token 倍数 |

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`。

**状态：** 未跑。当前实现是 ShortOPD 的 token-only structural detector；未实现
论文里的 OPD-loss/teacher-NLL onset refinement，结果中统一写
`ShortOPD-inspired`，不能写“复现 ShortOPD”。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

RUN22=Qwen3-1.7B-w2g128-shortopd-lr2e-6-wu30-ws2e-7-schhold-short2long100
test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

bash scripts/run_qwen3_1.7b_opd.sh --wbits 2 --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 100 --save-steps 5 \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --short-opd \
  --short-opd-min 1024 \
  --short-opd-max 8192 \
  --run-suffix short2long100

KEEP_CHECKPOINTS=1 \
EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${RUN22}" \
  --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
  --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${RUN22}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${RUN22}" \
  --out-dir "output/plots/${RUN22}"
```

**开训闸门：** 横幅必须显示 `mode=short-opd`、budget `1024..8192`；
`opd_step_metrics.jsonl` 每 step 必须包含：

- `opd/shortopd_budget`、`opd/shortopd_batch_budget`
- `opd/shortopd_repetition_rate`
- `opd/shortopd_clean_truncation_rate`
- `opd/shortopd_effective_length`

每个 Trainer checkpoint 还必须包含 `short_opd_state.json`，否则不能安全 resume。

**判断有效：** 与 `#15` 同 step 比较，不只看最终分数。至少同时报告
GSM8K/MATH-500、paper-5/full-suite 最佳点、累计 rollout tokens、训练墙钟、
repetition rate、budget 轨迹、EOS/truncation、grad norm 和 code jump。方法只有在
精度不降或提升的同时明显降低 rollout tokens/墙钟，或者以相同成本更快达到
`#15` 的分数，才算有效。若有信号，再补 fixed-H 对照以拆分“动态 controller”
与“单纯缩短 horizon”。

产出：

```
log/distill/${RUN22}/checkpoint-{5,10,...,100}
output/distill/${RUN22}/checkpoint-{5,10,...,100}
output/eval/${RUN22}/checkpoint-{5,10,...,100}
output/plots/${RUN22}/opd_timeline.{csv,jsonl}
```

---

## Exp #23（2026-09-18 新增）— Stage 1 → OPD 目标迁移与 W2 量化平台诊断

**目标：** 不重训，只复用已有 W2 Stage 1 和 `#10/#15` checkpoint，判断
OPD 前 30 step 的分数先降后升主要来自：

1. Stage 1 reconstruction 与 Stage 2 OPD 的目标冲突；
2. master weight 尚未跨过足够多 W2 量化边界形成的离散平台；
3. on-policy 数据分布变化，而不是固定目标本身改善。

本实验区分“目标迁移”和“局部最优”。若 Stage 1 起点的 fixed-OPD gradient
明显非零，就不能写成“陷入 Stage 2 局部最优”。

### Checkpoint

固定分析 13 个点：

```text
0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 85
```

| step | 来源 |
|---:|---|
| 0 | `output/block_qat/Qwen3-1.7B-w2g128` |
| 5–30 | `output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-*` |
| 35–85 | `output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-*` |

step 0–40 是主分析区间；50/60/80/85 只确认后期是否稳定。不要混入
ShortOPD checkpoint，也不要用 `#21` 的长续训回答 warmup 机制。

### 固定诊断数据

**Stage 1 原生 probe：** 使用原 Stage 1 的 `sweep_0.8` validation split，
`seed=2`、`val_size=64`、`sequence_length=2048`。所有 checkpoint 使用完全相同
的 token、mask 和 FP teacher activation。

**Common probe：** 对 `open-thoughts/OpenThoughts3-1.2M` 复现 Stage 2 的
`shuffle(seed=2)`，跳过已经用于训练的前 32768 条，取 index
`32768:32896` 共 128 个 prompt；按训练口径过滤 tokenized prompt length
`>=8192`。保存最终样本 ID/原始 index，所有 checkpoint 复用同一集合。

在 common probe 上只生成一次并缓存两套固定轨迹：

- Probe-A：由 W2 Stage 1 模型生成；
- Probe-B：由 warmup `checkpoint-30` 生成。

两套生成都使用当前 OPD 的 chat template、`T=0.6` 和总长度上限 8192。缓存
prompt tokens、response tokens、response mask；后续评估 checkpoint 时禁止重新
rollout。teacher、tokenizer 和 loss mask 必须固定。

### 必报指标

1. **Stage 1 block-mean reconstruction loss**：在每个 Transformer block 输出
   边界计算 quantized student 与 BF16 teacher 的 hidden-state MSE，再对 block
   等权平均。分别在 `sweep` 和 common probe 上报告。
2. **Stage 1 last-output loss**：完整 teacher/student 端到端前向后，在最后一个
   Transformer block 输出、final RMSNorm 之前计算有效 token 的 hidden-state
   MSE：

   \[
   L_{\mathrm{S1,last}}=
   \frac{\sum_{b,t}m_{b,t}\lVert h^{(L)}_{S,b,t}-h^{(L)}_{T,b,t}\rVert_2^2}
        {D\sum_{b,t}m_{b,t}}.
   \]

   同时报 raw MSE 和除以 teacher hidden-state 能量的 relative MSE；主表使用
   common probe，附表报告 `sweep`。它不是 logits loss，也不是 OPD loss。
3. **Fixed Stage 2 loss**：在 Probe-A/Probe-B 固定 token 上，按当前训练实现计算
   response-token-only sampled reverse-KL surrogate，得到 `fixed_opd_A/B`。禁止
   用每个 checkpoint 自己的新 rollout 横向比较。
4. **量化状态**：相对 Stage 1 起点的 master-weight relative change、累计
   quantized-code flip rate；相邻 checkpoint 的 code jump 和 code-jump
   amplification。累计 code flip 定义为：

   \[
   C_t=\frac{1}{N}\sum_i \mathbf 1[q_i(\theta_t)\ne q_i(\theta_0)].
   \]

5. **已有行为结果**：直接复用相同 checkpoint 已完成的 GSM8K、MATH-500，
   报 `Average=(GSM8K+MATH-500)/2`。不要重跑完整 benchmark；70/75/80/85 的
   full-suite 仅作补充。

所有 loss 除全局 token mean 外，还要保留 sample mean/标准误。不同目标的绝对
尺度不可直接比较，额外报告相对 step 0 的归一化变化：

\[
\Delta L_j(t)=\frac{L_j(\theta_t)-L_j(\theta_0)}
{|L_j(\theta_0)|+\epsilon}.
\]

### 主结果表

| step | S1 block mean | S1 last output | fixed OPD-A | fixed OPD-B | master change | cumulative code flip | GSM8K | MATH-500 | Average |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | | 0 | 0 | | | |
| 5 | | | | | | | | | |
| 10 | | | | | | | | | |
| 15 | | | | | | | | | |
| 20 | | | | | | | | | |
| 25 | | | | | | | | | |
| 30 | | | | | | | | | |
| 35 | | | | | | | | | |
| 40 | | | | | | | | | |
| 50 | | | | | | | | | |
| 60 | | | | | | | | | |
| 80 | | | | | | | | | |
| 85 | | | | | | | | | |

主表填写 common-probe loss；`sweep` 的 block-mean/last-output loss 和逐层结果
放附表。至少绘制：

1. `S1 block mean / S1 last output / fixed OPD-A/B` 对 step；
2. `master change / cumulative code flip` 对 step；
3. `cumulative code flip / GSM8K / MATH-500` 的时间对齐图。

### 可选梯度诊断

只有 loss 曲线显示冲突时，才在 step `0/15/30/40/80` 补算
`||g_S1||`、`||g_OPD||` 和 gradient cosine；按 layer 累计 dot product，不保存
整模型双份梯度。

### 判定规则

- `S1 common loss ↑` 且 `fixed OPD loss ↓`：支持目标冲突；若 last-output 同时
  上升，说明冲突已经传到最终 hidden state。
- block-mean 基本不变而 last-output 上升：平均局部误差掩盖了末端累计误差。
- block-mean 上升但 last-output 不升：更像内部表示重组，不能声称 Stage 1
  能力被破坏。
- master weight 先变化、累计 code flip 后出现，随后 fixed OPD loss/benchmark
  才改善：支持 W2 量化平台。
- 两种 Stage 1 loss 与 fixed OPD loss 同时下降：不支持目标冲突，warmup 更可能
  主要控制优化噪声和量化码翻转。
- fixed OPD loss 下降但 benchmark 不恢复：token-level 拟合改善尚未转化为推理
  能力。

**状态：** 待做离线统计；不启动 Stage 1/Stage 2，不重新生成每个 checkpoint
的 rollout，不重新跑 GSM8K/MATH-500。

产出：

```text
output/analysis/exp23_stage_transition/probe_manifest.json
output/analysis/exp23_stage_transition/checkpoint_metrics.{csv,jsonl}
output/analysis/exp23_stage_transition/layer_metrics.csv
output/analysis/exp23_stage_transition/*.png
```

---

## Exp #24（2026-09-21 新增）— W2 PV-OPD RawGate：warm30 + stable70

**目标：** 从 Exp #4 的 W2 Stage 1 权重重新起训，在与 Vanilla OPD 相同的
`30 step warmup + 70 step hold` 日程下，验证不做 batch 均值归一化的
token-level precision gate。不要复用 `#10 checkpoint-30`：Gate 从 step 1
开始改变梯度，复用 Vanilla warmup 只能回答 tail-only 问题。

W2 Student 负责 rollout；冻结的 BF16 Teacher 和共享当前 master weights、
group size、实数 clipping range 的 W4 Probe 在相同 W2 history 上打分：

\[
A_t^{FP}=\log p_T(a_t\mid h_t^2)-\log p_2(a_t\mid h_t^2),\qquad
A_t^{Prec}=\log p_4(a_t\mid h_t^2)-\log p_2(a_t\mid h_t^2).
\]

使用原始 Gate，不做 per-batch 或 fixed-warmup normalization：

\[
g_t=\mathbf 1[A_t^{FP}A_t^{Prec}>0]
\min\left(1,\frac{|A_t^{Prec}|}{|A_t^{FP}|+\varepsilon}\right),
\qquad g_t\in[0,1].
\]

目标函数为：

\[
L_{PV}=\frac1N\sum_t
\operatorname{sg}\!\left[g_t\operatorname{clip}(-A_t^{FP},-c,c)\right]
\log p_2(a_t\mid h_t^2).
\]

`c` 继续由前 10 optimizer steps 的 `|A_FP|` P99 标定；本实验只取消 Gate
normalization，不同时移除 advantage clipping、增加 gate floor 或改变 loss。
CE=0，W4 Probe 只验证信号且全程 `no_grad`。更新 W2 master weights、非量化
权重和 quantizer scale；冻结 zero point、embedding、Teacher 和 Probe。

### 固定设置

| 项 | 设置 |
|---|---|
| 初始化 | `output/block_qat/Qwen3-1.7B-w2g128`（Exp #4 Stage 1） |
| 数据 | OpenThoughts requested 32768，`seed=2`，自动 prompt-only 预过滤 |
| 长度 | `max_length=8192`，`min_rollout_tokens=1` |
| 量化 | W2 target / shared-range W4 probe / group size 128 |
| Batch | 8 GPU，per-device 1，effective batch 64 |
| LR | step 0 为 `2e-7`，30 steps 升到 `2e-6` |
| Stable | step 31–100 使用 `constant_with_warmup` hold `2e-6` |
| Loss | RawGate × clipped sampled reverse-KL，CE=0，temperature=0.6 |
| 保存 | 每 5 optimizer steps |
| 评测 | 只评 GSM8K、MATH-500 |

启动前必须看到 prompt filter 的 `before/after/filtered/max_before/max_after`，且
`max_after<=8191`。正式对照必须是使用相同修复后数据口径的 Vanilla
`warm30 + stable70`；基于错误 2124 条样本的数据轨迹不能作为正式结果。

**依赖：** `output/block_qat/Qwen3-1.7B-w2g128/config.json`。不依赖任何
Vanilla OPD Trainer checkpoint。

**状态：** RawGate 代码与启动参数已加入；token-level 记录协议已确定但代码尚未
实现。完成下面的记录与 1-step smoke 前不要启动正式 GPU 训练；当前尚无结果。

### Token-level 诊断记录（正式开跑前的硬要求）

记录所有 rank、所有 gradient-accumulation micro-batch 中的**有效 rollout
response token**；位置从每条 response 的 0 开始。prompt、padding 和 EOS 后区域
不记入。诊断张量全部 `detach`，只用于离线分析，不参与 loss，也不得改变训练
轨迹。

每个 token 至少保存：

```text
step, sample_id, response_pos, token_id,
student_logp, teacher_logp, probe_logp,
A_fp, A_prec, same_direction, gate, clipped_advantage, is_eos
```

每条 trajectory 额外保存 `prompt_length`、生成 token ids、`response_length`、
`hit_max_length`、`repeated_suffix`，以及在可可靠自动判分时的 `is_correct`；不能
可靠判分的 OpenThoughts 样本保留为空，不能用 Teacher 分数冒充正确性。按
optimizer step 写紧凑 `.pt`，训练后再合并：

```text
log/distill/<tag>/pv_token_traces/step-<step>-rank-<rank>.pt
```

不保存完整 vocabulary logits。若选定少量异常 checkpoint 后确实需要候选分布，
再对这些 checkpoint 单独补 top-k logits。1-step smoke 必须确认 8 个 rank 文件均
存在、每个 `response_pos` 从 0 连续递增，且所有文件的有效 token 数之和等于训练
日志中的有效 response token 数。

```bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate reasoningqat

test -f output/block_qat/Qwen3-1.7B-w2g128/config.json

CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/test_pv_opd.py
bash -n scripts/run_qwen3_1.7b_pv_opd.sh

bash scripts/run_qwen3_1.7b_pv_opd.sh \
  --stage 2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --probe-bits 4 \
  --gate-mode full \
  --gate-normalization none \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup \
  --max-steps 100 \
  --save-steps 5

TAG=Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold

KEEP_CHECKPOINTS=1 EVAL_DATASETS="gsm8k math_500" \
  bash scripts/eval_distill_checkpoints.sh \
  --watch-dir "output/distill/${TAG}" \
  --wbits 2 --eval-gpu 0 \
  --steps 5,10,15,20,25,30,35,50,75,100 --skip-final

python scripts/merge_opd_timeline.py \
  --metrics "log/distill/${TAG}/opd_step_metrics.jsonl" \
  --eval-root "output/eval/${TAG}" \
  --out-dir "output/plots/${TAG}"

# 转换 final 时必须重复影响 TAG 的参数。
bash scripts/run_qwen3_1.7b_pv_opd.sh \
  --stage 3 \
  --gate-normalization none \
  --lr 2e-6 \
  --warmup-steps 30 \
  --warmup-start-lr 2e-7 \
  --lr-scheduler constant_with_warmup
```

训练启动后先检查：

1. 横幅包含 `gate=full normalization=none`、`lr=2e-6`、
   `warmup_steps=30`、`scheduler=constant_with_warmup`。
2. `pv/gate_mean` 是 RawGate 的实际均值，不应被强制到约 1；
   `pv/gate_keep_rate`、`pv/same_direction_rate`、`pv/adv_clip` 正常落盘。
3. step 30 的 LR 达到约 `2e-6`，step 35/50/75/100 保持约 `2e-6`。
4. `opd_step_metrics.jsonl`、`opd_timing_summary.json`、timeline 与训练墙钟按
   文末结果清单报告；不要只报最终分数。

### 预期结果与判断路径

本实验的首要预期不是保证最终分数上涨，而是验证 W4 Probe 能否在 W2 rollout
上识别可信的 Teacher 修正方向。RawGate 未归一化，故 `gate_mean` 应自然低于
batch-normalized 版本；同 LR 下有效更新通常更小。若假设成立，预期表现为：

1. warmup 初期可能比 Vanilla 涨得慢，但 code jump、重复后缀和 response 顶满率
   更低；stable 段继续提升，而不是在少数 checkpoint 偶然冲高。
2. 同方向且 Gate 较高的 token 更集中在非重复、正常 EOS 的 trajectory；在能自动
   判分的样本上，它们应更偏向正确 trajectory。这里只能证明相关性，不能直接
   声称这些 token 具有因果贡献。
3. 主结果看 step 35/50/75/100 的
   `(GSM8K + MATH-500) / 2` 曲线面积与 final，不以单个 best checkpoint 下结论。
   暂定成功线：相对 clean Vanilla 至少连续两个 checkpoint 的平均分提高
   `>=1.0`，且 final 不退化；同时没有以明显更长 response 或更高顶满率换分。

训练结束后先按 `response_pos` 统计 `same_direction_rate`、`gate_mean`、
`|A_fp|`、`|A_prec|`，再分别按正确/错误、重复/非重复、EOS/顶满 trajectory
分组。随后只走一个分支：

| 观察 | 下一步 |
|---|---|
| 分数稳定提升，且高 Gate 与正常/正确轨迹相关 | 跑 batch-normalized PV 对照，再做 shuffled-gate，确认收益不是单纯缩小梯度 |
| 训练更稳但分数不涨，`gate_mean` 很小 | 保持 Gate 排序，只做等均值 rescale 对照；不要同时改 `c` |
| 高 Gate 与正确性无关，shuffled-gate 也相当 | Gate 没提供 precision 信息，停止扩展该方向 |
| `same_direction_rate` 很低，W4 经常与 Teacher 反向 | W4 Probe 不够可靠；下一次只比较更高 probe bit 或独立校准，不继续调 Gate 公式 |
| 重复/顶满集中在后段且对应异常 Gate | 再做 position/repetition mask 消融；先不加入主实验 |

“哪些 token 有用”的最终结论必须由后续 token-mask 或 shuffled-gate 消融验证；
本实验的 token-level 统计只负责提出候选规律。

主比较使用相同 step 的 GSM8K/MATH-500、grad norm、code jump、Gate 均值和
rollout 时间。若 RawGate 训练稳定且优于 clean Vanilla，再补完全相同 schedule
的 batch-normalized PV 消融；本实验不预先支付第二条 PV 训练成本。

产出：

```text
output/distill/Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold
output/eval/Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold
log/distill/Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold
log/distill/Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold/pv_token_traces
output/plots/Qwen3-1.7B-w2g128-pv-opd-rawgate-lr2e-6-wu30-ws2e-7-schhold
```

---

## Exp #26（2026-09-23 新增）— W2 Vanilla OPD 短上下文 / 短 rollout 消融

**目标：** 从 Exp #4 的 W2 Stage 1 权重重新起训，检验把 prompt + response
总长上限由 `8192` 缩短到 `2048`、response 上限设为 `1024` 后，能否明显降低
rollout 成本并保留 GSM8K / MATH-500 表现。这是长度—成本消融，不是与 8192
基线等价的训练。

**保持不变：** W2A16、group size 128、同一 Stage 1、数据顺序与 seed、sampled
reverse-KL、CE=0、temperature=0.6、8 GPU、effective batch 64。使用 stable
日程：前 30 step 从 `2e-7` 线性 warmup 到 `2e-6`，随后 120 step 恒定保持
`2e-6`，共 150 optimizer steps。

| 项 | 8192 基线 | Exp #26 |
|--|--:|--:|
| prompt + response 总长上限 | 8192 | **2048** |
| response 上限 | 剩余总长动态决定 | **1024** |
| 每个 batch 的实际 response budget | `8192 - padded_prompt_len` | **`min(1024, 2048 - padded_prompt_len)`** |

**实现约束：** 当前 Vanilla OPD 会将 `max_new_tokens=None`，只用 `max_length`
控制总长；因此本实验启动前需要显式实现 response cap。不能简单地同时把
`max_length=2048` 和 `max_new_tokens=1024` 交给 `generate`：
`max_new_tokens` 可能优先生效，使总长超过 2048。代码必须逐 batch 计算上表中的
实际 budget。

prompt-only 预过滤仍只保证至少 1 个 rollout token，即 `prompt_length <= 2047`。
因此长 prompt 能生成的 response 可能少于 1024。若要求每条样本都保留 1024-token
response 空间，则应过滤为 `prompt_length <= 1024`，但这会额外改变 prompt 分布，
本实验暂不采用。

**必报结果：** 与 8192 基线按相同 optimizer step 对齐，报告 GSM8K、MATH-500、
平均 response length、截断率、EOS 率、累计 rollout tokens、rollout 墙钟时间、
sec/step 和总训练时间。若分数下降，需要区分长 prompt 被压缩、response 被 1024
cap 截断，以及有效 loss token 减少这三种原因。

**判定：** 若两项分数不明显退化，同时累计 rollout tokens 和墙钟时间明显下降，
则支持把短 rollout 用作前期训练或自适应长度起点；否则说明 W2 OPD 仍需要长轨迹。

**依赖：** Exp #4 Stage 1。训练参数尚未实现，确认前不启动 GPU 训练。

**状态：** 未跑；仅完成实验设计。

---

## Exp #27（2026-09-23 新增）— W3 Vanilla OPD 短上下文 / 短 rollout 消融

**目标：** 做 Exp #26 的 W3 对应实验，但保留 W3 自己的训练超参数。以 Exp #2
的 W3 Vanilla OPD 为对照，只把 prompt + response 总长上限从 `8192` 改为
`2048`，并把 response 上限设为 `1024`，检验短 rollout 在 W3 上的速度—性能
权衡。该实验与 Exp #26 共同观察低 bit 是否比 W3 更依赖长推理轨迹。

**保持不变：** W3A16、group size 128、Exp #1 的同一份 W3 Stage 1、
OpenThoughts 顺序与 seed、sampled reverse-KL、CE=0、temperature=0.6、8 GPU、
effective batch 64、不训练 embedding。使用 stable 日程：前 30 step 从 `1e-7`
线性 warmup 到 W3 峰值 `1e-6`，随后 120 step 恒定保持 `1e-6`，并显式固定
`max_steps=150`，避免 2048 预过滤改变数据集大小后连带改变 optimizer step 数。

| 项 | W3 8192 基线 | Exp #27 |
|--|--:|--:|
| prompt + response 总长上限 | 8192 | **2048** |
| response 上限 | 剩余总长动态决定 | **1024** |
| 每个 batch 的实际 response budget | `8192 - padded_prompt_len` | **`min(1024, 2048 - padded_prompt_len)`** |
| optimizer steps | 150 | **150** |
| 最大 prompt presentations | 9600 | **9600** |

**实现与过滤约束：** 与 Exp #26 相同，启动前必须显式实现逐 batch 的 response
cap；不能直接依赖 Hugging Face 同时设置 `max_length` 和 `max_new_tokens`。
prompt-only 过滤条件保持 `prompt_length <= 2047`，并在状态中记录
`before / after / filtered / max_before / max_after`。150 steps 最多消费 9600 个
prompt；若过滤后不足 9600 条就会重复采样，须报告 unique prompt 数和重复比例。

**必报结果：** 与 W3 8192 基线按相同 step 比较 GSM8K、MATH-500、平均 response
length、截断率、EOS 率、累计 rollout tokens、rollout 墙钟时间、sec/step 和
总训练时间。同时将 Exp #26 / #27 按相同 step 对齐，比较 W2 与 W3 的性能下降和
成本下降比例。

**判定：** 若 W3 在 2048/1024 下基本保持分数，而 W2 明显退化，支持“量化容量
越低，越依赖长轨迹或更完整的 token-level 纠正”；若 W2/W3 都保持性能，则优先
使用短 rollout 降低 OPD 成本；若两者都退化，则主要是任务长度预算不足。

**依赖：** Exp #1 W3 Stage 1，以及与 Exp #27 日程完全一致的 W3 8192 对照。
训练参数尚未实现，确认前不启动 GPU 训练。

**状态：** 未跑；仅完成实验设计。

---

## Exp #28（2026-09-26 新增）— W2 8192-rollout 末尾重复监督诊断

**问题：** Vanilla OPD 的长 response 中，有多少监督 token 实际落在末尾重复区？
先做长度与占比诊断，不把“发生重复”直接等同于“监督无效”。

固定 `max_length=8192`、temperature `0.6`、`top_k=0`，从 OpenThoughts train
按训练口径 `shuffle(seed=2)` 后取前 64 个可生成 prompt。所有 checkpoint 使用
完全相同的 prompt、顺序和 sampling seed；每次只生成一条轨迹，匹配训练时的
on-policy 单样本监督。这里是在各 checkpoint 上重新采样的固定 probe rollout，
不是训练时已经消费过但未保存的历史 rollout；它用于比较不同 checkpoint 的生成
行为，不能还原某一步实际见过的随机轨迹。分析以下 11 个点：

| step | checkpoint |
|---:|---|
| 0 | `output/block_qat/Qwen3-1.7B-w2g128` |
| 5–30 | `output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-*` |
| 35–50 | `output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-*` |

检测到的末尾周期重复沿用现有 ShortOPD 口径（period 1–10、末尾 anchor 一致率
`>=0.9`、至少 64 tokens/3 cycles）；离线统计回溯完整 response，不受训练时
512-token 检测窗口限制。该长度是 90% 一致率下的近似区间，不视为人工标注的
精确重复起点。每个 step 必报：

- `repetition_rate`；
- 平均 response 长度；
- 平均末尾重复长度（全样本，无重复记 0）及重复样本条件均值；
- 平均重复前长度 `response_length - terminal_repeat_length`，以及重复样本条件均值；
- `repeat_token_fraction = sum(repeat_length) / sum(response_length)`；
- EOS rate 与 hit-max-length rate。

```bash
python scripts/analyze_rollout_repetition.py \
  --checkpoint 0=output/block_qat/Qwen3-1.7B-w2g128 \
  --checkpoint 5=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-5 \
  --checkpoint 10=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-10 \
  --checkpoint 15=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-15 \
  --checkpoint 20=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-20 \
  --checkpoint 25=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-25 \
  --checkpoint 30=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold/checkpoint-30 \
  --checkpoint 35=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-35 \
  --checkpoint 40=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-40 \
  --checkpoint 45=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-45 \
  --checkpoint 50=output/distill/Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold-from30hold/checkpoint-50 \
  --max-length 8192 --num-prompts 64 \
  --output-dir output/analysis/exp28_rollout_repetition
```

主表：

| step | repetition rate | response len | repeat len | repeat len (repeated) | pre-repeat len | pre-repeat len (repeated) | repeat-token fraction | EOS rate | hit-max rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | | | | | | |
| 5 | | | | | | | | | |
| 10 | | | | | | | | | |
| 15 | | | | | | | | | |
| 20 | | | | | | | | | |
| 25 | | | | | | | | | |
| 30 | | | | | | | | | |
| 35 | | | | | | | | | |
| 40 | | | | | | | | | |
| 45 | | | | | | | | | |
| 50 | | | | | | | | | |

**判断：** 若 repeat-token fraction 在早期 checkpoint 很高，说明 OPD 的有效
token 预算被重复尾部大量占用，但仍不能仅凭长度断言这些梯度无效。下一步只做一组
因果消融：训练时 mask 掉检测出的重复尾部，并与“随机 mask 相同 token 数”的对照
比较；前者显著更好才支持“重复 token 监督有害”，持平只支持“冗余”。

**状态：** 统计脚本与记录格式已完成，尚未启动 GPU rollout。

产出：

```text
output/analysis/exp28_rollout_repetition/probe_manifest.json
output/analysis/exp28_rollout_repetition/checkpoint-*.jsonl
output/analysis/exp28_rollout_repetition/summary.{csv,json}
```

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

**强制：** 每一次 OPD / Stage 2 训练结束后，都要把下面这套 log **收齐、合并、画图**，再在该 Exp 的状态里挂路径。只报最终 GSM 分、不交 timeline 图 / 时间数字，算没做完。

### 训练时自动落盘

Stage 2 开 `--opd` 时，rank0 自动写：

| 文件 | 内容 |
|------|------|
| `log/distill/<tag>/opd_step_metrics.jsonl` | 每 optimizer step 一行 |
| `log/distill/<tag>/opd_timing_summary.json` | 训完汇总：wall / mean step / mean rollout |

字段（前缀 `opd/` 的也进 Trainer log）：

- **truncation：** `truncation_rate`、`eos_rate`、`mean_response_len`、`hit_max_length_rate`（无 EOS 且顶满 `max_length` 算截断）
- **跳码：** `code_jump_rate`、`code_jump_count`、`master_rel_change`、`code_jump_amplification`
- **时间：** `rollout_seconds`（该 step 内 generate 墙钟，含 grad accum 微批求和）、`step_seconds`（相邻 optimizer step 墙钟）
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

可选加面板：`opd/rollout_seconds`、`opd/step_seconds`。

图保存到 `output/plots/<tag>/`（如 `timeline.png` / 分面板 PDF）。状态栏示例：

```text
metrics: log/distill/<tag>/opd_step_metrics.jsonl
timing: log/distill/<tag>/opd_timing_summary.json
timeline: output/plots/<tag>/opd_timeline.csv
plots: output/plots/<tag>/timeline.png
```

---

## 结果报告清单（后续 Exp 强制）

结案时在该 Exp **状态**里同时写 **分数 + 时间**，不要只贴 GSM/MATH。硬件与设定也要一行写清（GPU 型号×卡数、`max_length`、effective batch、总 step）。

### 必报（所有 Stage 2：GKD / ReasoningQAT / OPD）

| 项 | 从哪读 | 备注 |
|--|--|--|
| 主评测分 | `output/eval/<tag>/` | 公共约定套件；至少写出主表五任务 + GSM8K |
| Stage 2 总墙钟 | 脚本起止 / `opd_timing_summary.json` 的 `wall_clock_seconds` / HF `train_runtime` | 写成小时或分钟，注明是否含 eval |
| 总 optimizer steps | 横幅 / `trainer_state.json` | 例如 512 / 100 |
| steps/hour 或 sec/step | 总墙钟 ÷ steps | 便于和别的 Exp 比吞吐 |
| 硬件 | `nvidia-smi` | 如 8×H20，`max_length=8192` |

非 OPD（`#9` / `#11` 等）若没有 `opd_timing_summary.json`：用训练 log 起止时间，或 `log/distill/<tag>/trainer_state.json` / 最终 log 里的 `train_runtime`、`train_samples_per_second`。

### OPD 额外必报

| 项 | 从哪读 |
|--|--|
| 平均 rollout 时间 / step | `opd_timing_summary.json` → `mean_rollout_seconds` |
| 平均 step 墙钟 | 同上 → `mean_step_seconds` |
| rollout 占 step 比例 | 同上 → `rollout_fraction_of_step` |
| 截断率走势 | 图或 JSONL 摘要（起 / 中 / 末） |

状态示例：

```text
eval: GSM8K=.. MATH-500=.. (full suite path=...)
timing: Stage2 wall=3.2h (8xH20), 100 steps, 115s/step;
        mean_rollout=98s (85% of step); summary=log/distill/<tag>/opd_timing_summary.json
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
（须附：benchmark 分 + Stage2 墙钟/sec-per-step；OPD 另附 rollout 时间、
 opd_step_metrics.jsonl、opd_timing_summary.json、opd_timeline、plots）

```bash
# 只写这一次的命令
# 训完：merge_opd_timeline.py + 出图 → output/plots/<tag>/
# 状态里填结果报告清单（分数 + 时间）
```
```
