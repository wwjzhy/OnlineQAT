# OPD / Ultra-Low-Bit QAT 论文主线（给 Codex 讨论用）

写于 2026-09-13。这是给 **Codex / 另一个模型** 的独立简报，不替代 `Exp.md`。数字以训练集群上的 log 为准；本机 `/zju_0038` 往往没有完整产物。

讨论时请先读完「能主张 / 不能主张」和「现在够不够投稿」。不要把两套设定合成一篇论文。

---

## 1. 一句话主张（当前主线）

**极低 bit（W2/W3）量化恢复里，offline QAT 在冻结前缀上对齐教师，部署却是学生自己的轨迹。我们检验：在学生自己的 occupancy 上做 on-policy distillation（OPD），是否比 matched offline QAT 更能恢复长推理。**

方法不新（GKD / DAgger / on-policy KD）。贡献应是：**问题设定（量化恢复里的 occupancy 错位）+ 极低 bit 上的证据 + W2 格子放大更新的机制**，不是「发明了 on-policy distillation」。

不要写成：「我们发现了量化特有的全新 exposure bias」。
更准确：**quantization-amplified occupancy mismatch** / **autoregressive policy drift after quantization**。

---

## 2. 两套线，不要混

| | 旧稿 `OPT-QAT/arxiv/` | **当前主线 `OnlineQAT/`** |
|--|--|--|
| 模型 | Qwen3-4B / 8B（表是空的） | **Qwen3-1.7B**（#14 才扩 4B 离线） |
| 位宽 | W4A16 | **W3 / W2，g=128** |
| thinking | 关；greedy；max 512/2048 | **开**；T=0.6 / top_k=20 / **8192**（evalscope） |
| 主指标 | GSM / MATH / HumanEval / MBPP | 论文主表五任务：MATH-500、LiveCodeBench、MMLU-Redux、GPQA-Diamond、IFEval（W3 论文约 **55.2**）；GSM/AIME 顺带 |
| 旧稿结论 | 甚至把 W2/W3 写成 future work | 现在全部证据都在 W2/W3 1.7B |

**不要**把 #10 的 GSM 填进旧 INT4 表。投稿只认当前这条 ultra-low-bit 线。目标会议口径曾写 ICLR 2027，以用户最终决定为准。

---

## 3. 理论（引用，不新造定理）

自回归 LM 是策略 \(\pi(\cdot\mid s_t)\)，\(s_t=(x,y_{<t})\)。部署活在学生占用 \(d_{\pi_Q}\) 上，不活在冻住的语料上。

- **Offline QAT / teacher forcing** = 在冻结占用 \(d_{\mathrm{off}}\)（金标或教师生成后冻住）上做 BC。Ross & Bagnell 2010：部署代价 \(O(T^2\varepsilon)\)，\(\varepsilon\) 只在专家/数据占用上小。
- **OPD** = 在 \(d_{\pi_Q}\) 上蒸馏。DAgger：误差若在学习者自己的占用上小，界变成 \(O(T\varepsilon)\)（或 \(uT\varepsilon\)）。\(u=\Theta(T)\) 时线性优势消失，必须写进文里。
- 量化网格 \(\Delta_b=\Theta(2^{-b})\)。未训练学生、Lipschitz 下 \(\varepsilon\) 可随 bit 变大；这是**上界动机**，不是已证「W2 一定 \(T^2\) 更紧」。
- Pinsker 只保证在**你拿来算 KL 的那个占用**上 TV 小。offline 把 data-KL 做低，**不蕴含** free-running KL 低。

**理论能说：** offline 目标和自由生成重建不一致；OPD 与「在学生轨迹上贴近教师」一致。  
**理论不能说：** OPD 的 GSM / 五任务均值一定更高。精度谁高，实验判。

---

## 4. 方法（当前实现）

两阶段，都在 `OnlineQAT/`。

**Stage 1（block QAT，`main_block_qat.py`）**  
- 校准 + 把权重重建进 W2/W3 网格（`sweep_0.8`：OpenThoughts 80% + FineWeb 20%）。  
- W3 `weight_lr=1e-5`；W2 `2e-5`；quant scale lr `1e-4`；g=128。  
- **不训练 embedding**。  
- 作用：给学生一块还能生成的格子，不是 occupancy 方法本身。

**Stage 2 OPD（`main_e2e_distill.py` + `scripts/run_qwen3_1.7b_opd.sh`）**  
- 学生 on-policy rollout，再在该轨迹上 **sampled reverse KL**，**CE=0**。  
- 教师 = 同一份冻结 BF16。  
- 数据：OpenThoughts 32768；effective batch **64**；`max_length=8192`；8 卡。  
- W2 稳定配方：峰值 LR **`2e-6`**，warmup 30 step 从 `2e-7` 升到峰值，之后应 **hold**（见 #10/#15）。

**Stage 2 离线对照（必须有，否则没有主贡献）**  
- 论文目标：\(L=0.2\,L_t+1.0\,\mathrm{KL}(\pi_T\Vert\pi_S)\)，\(L_t\) 是 teacher-weighted CE，`forward_kl`，cosine。  
- 脚本：`run_qwen3_1.7b_reasoningqat.sh`（#9 W3，#11 W2）。  
- **不要**把旧 GKD（plain CE + JSD，`run_qwen3_1.7b.sh`）当成论文离线主方法。那是错 loss。

和 BitDistiller 的刀口：他们主要在**冻住的教师生成序列**上蒸馏，并报告 teacher-generated 优于 student-generated。我们做的是**训练中持续更新的学生在线轨迹**。必须在文中写清，否则审稿人会拿他们的消融打过来。

---

## 5. 评测协议（当前主线）

`scripts/eval_paper_benchmarks.sh`：vLLM + evalscope `openai_api`。  
`T=0.6`，`top_k=20`，`max_tokens=8192`。Qwen3 **thinking on**（`ENABLE_THINKING=1`）。

不要和 OPT-QAT 早期协议比绝对分：那是 thinking **off**、greedy、`max_new_tokens=2048`。

---

## 6. 实验地图与状态

| Exp | 内容 | 状态（2026-09-13） |
|--|--|--|
| #1 | W3 离线旧 GKD（plain CE+JSD） | 流程在；**不能当论文 offline** |
| #2 | W3-OPD | 有 run；逐步 JSONL **缺**，#13 只截前 50 step |
| #3 | 只评 W3 Stage 1 | 基线用 |
| #4 | W2 Stage 1 + 旧 GKD | Stage 1 是后面 W2-OPD 的起点 |
| #5 | W2-OPD，默认 `5e-6`，≤100 step | **崩**：GSM 25.4→0.5(s20)→22.7(s35)；grad 43→172；jump 放大 20–44× |
| #6 | PV-OPD | 先不要并进主文 |
| #7 | W2-OPD `1e-6`，50 step | 步长消融 |
| #8 | W2-OPD `2e-6`+wu30，50 step，linear 衰减 | GSM 约 25→40，还在涨 |
| #9 | W3 **论文** ReasoningQAT | **未跑 / 主表未填** |
| #10 | W2-OPD `2e-6`+wu30，100 step，**本应 hold** | **已跑完，但是 decay 不是 hold**（见下） |
| #11 | W2 论文 ReasoningQAT | **未跑 / 主表未填** |
| #12 | 离线六档时间/收敛表 | 未填 |
| #13 | W2-OPD 100 vs W3-OPD 50 诊断 | W2 有 JSONL；W3 无 |
| #14 | Qwen3-4B 复现离线六档 | 未跑；不要先做 4B-OPD |
| #15 | 从 #10 的 **ckpt-30** hold `2e-6` 跑到 100 | **未跑**；不要从 step 100 续 |
| #16 | BF16 1.7B **thinking on** GSM 上界 | **未跑** |

脚本默认 8 卡 Stage 2。这台 4 卡机（`ajv59d3mbdufs-0`）经常被占满，**不要抢卡**。真训在另一台 8 卡 H20。

---

## 7. 已经钉死的数字

### 7.1 Teacher / PTQ（OPT-QAT，thinking **off**，不能当 #10 上界）

- Qwen3-1.7B BF16：GSM **75.7%**（999/1319），MATH-500 60.6%，AIME-120 8.3%。路径：`OPT-QAT/outputs/eval_qwen3_1p7b_s0_bf16`。
- 同设定 W3 RTN（不训练）：GSM **0.53%**，MATH/AIME **0**。
- W3 GPTQ/AWQ（packed calib）：GSM 约 11–12%，仍远低于 BF16。

### 7.2 Exp #10（thinking on 协议；W2-OPD；**实际 linear→0**）

设计：`--lr 2e-6 --warmup-steps 30 --warmup-start-lr 2e-7 --lr-scheduler constant_with_warmup --max-steps 100`。  
实现 bug：`str(SchedulerType)` 匹配失败，warmup 完走了 linear。横幅仍打印 `constant_with_warmup`。已在 commit `0677a18` 用 `enum.value` 修好。**已跑完的 #10 分数必须标成 linear-100，不是 hold。**

实测日程：step 1–30 为 `2e-7→2e-6`（对）；31–100 为 \(2\times10^{-6}\times(1-(t-30)/70)\)（s61≈1.14e-6，s99≈5.7e-8）。

诊断（#13 已看 W2 侧）：

- GSM：**24.6% → 56.1%（s95）→ 54.8%（s100）**。
- 五任务均值：**几乎走平**（GSM 单点涨，主表没抬起来）。
- `truncation_rate` 全程 **0**；平均回复 **158–163** token。不是策略塌、不是顶满 8192。
- `grad_norm` **0.04–0.18**，无 #5 那种尖峰。
- loss **5.6 → 2.4**。
- `code_jump_amplification` 前期约 **20×**，后期约 **10×**（后期变小主要因为 LR→0）。
- rollout 占 step 时间 **约 88.6%**；墙钟约 45h / 100 step，约 26 min/step。
- 三次启动（`--resume`）：1–61，62–71，72–100；中间因过长 prompt 崩过，skip 修复在 `3a6bc55`。

产出 tag：`Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold`。  
#15 必须从 **`log/distill/<tag>/checkpoint-30`** 续，不要 step 100，不要覆盖 #10。

### 7.3 步长故事（已结束，不要再扫 LR）

- `#5` `5e-6`：振荡崩盘。master 只动 0.15–0.6%，码跳 6–12%（20–44×）。W2 只有 4 个码字。
- `#7` `1e-6`：更稳、涨得慢。
- `#8` `2e-6`+wu30、50 step：25→40。
- 结论：**默认崩是步长，不是 OPD 接线错误。** 不要再扫 `5e-6/1e-6/2e-6`。

---

## 8. 正在研究的机制

### 8.1 Stage 2「先掉点再提点」

两种曲线不要混：

1. **#5 病理振荡**：25→0.5→22。格子 + 大步长 + on-policy 反馈。
2. **软掉点再涨**：Stage 1 还行 → Stage 2 前几十 step 掉 → 再涨。更像离开 Stage 1 盆地。

进 OPD 同时换了三样：目标（recon → CE=0 + reverse KL）、占用（金标前缀 → 差的 \(\pi_Q\)）、W2 跳码（STE 放大）。掉点可以不是 bug。

#10 后半涨到 56% **不能**直接说「适应成功」：从 step 30 起 LR 在往 0 掉。#15 用来拆开。

隔离实验（尚未立项写入 Exp.md，讨论后可写成 #17）：

- 同 LR 的离线 GKD（不 rollout）：掉点变浅 → 锅在换目标，不只是 OPD。
- OPD 但冻住 scale/clip：掉点变浅 → 锅在跳码。
- 已有 #10 的 s5/s10/…/s30 必须先画出来，和 jump、grad、lr 对齐。

### 8.2 能不能不要 Stage 1、直接 OPD？

**可以当对照，不能当 W2 默认配方。**

Stage 1 校准 scale/clip 并把权重重建进网格。OPD 需要学生还能 roll 出教师打得了分的轨迹。W3 RTN 已经 GSM 0.53%：从垃圾占用上做 reverse KL 是冷启动失败，和「GRPO 要先 SFT」同类。

| 做法 | 预期 |
|--|--|
| BF16 套默认 fake-quant 直接 OPD | W2/W3 几乎起不来 |
| 只 minmax/GPTQ、不训 block QAT，再 OPD | 值得试，尤其 W3 |
| Stage 1 + OPD（现在） | 已证明 GSM 能从 ~25 到 ~56 |

若 naive+OPD 能涨：Stage 1 可后置，方法更干净。  
若只有 Stage1+OPD 涨：诚实写 **OPD 修占用，不负责从崩塌网格冷启动**。

短探针：先在 **同一 thinking 协议** 下评 naive W2/W3 的 step-0；W3 上 50–100 step 无 Stage 1 OPD；W2 若试，先冻 scale。不要用这个替换 #15。

---

## 9. 现在够不够投稿？

**不够。** #10 是稳定性证据，撑不起 occupancy 主文。

缺三块，缺一块主贡献都不成立：

1. **Matched offline**：#9 / #11 必须有。没有就不能写 better than offline QAD。旧 GKD 不算。
2. **主指标**：五任务均值，不是 GSM。#10 五任务走平。headline 不能只报 56% GSM。日程还是 decay。
3. **机制图**：offline vs on-policy KL、随 token 位置是否发散、首次分叉。现在都是 todo。jump 20× 只说明 2bit 格子放大更新，没证明「offline 修错了占用」。

最小主表（1.7B）：

| | W3 | W2 |
|--|--|--|
| BF16 thinking（#16，最好不只有 GSM） | 上界 | 上界 |
| Stage 1 | 能评 | 能评（#10 起点 ~24.6 GSM） | 
| 论文 offline QAT（#9 / #11） | 必须 | 必须 |
| OPD **真正 hold**（#15 + 对齐的 W3-OPD） | 必须 | 必须 |

过线：

- OPD 五任务均值明显高于同 bit 的 #9/#11，且至少 W2 或 W3 差距不像噪声 → 主文成立。
- 只赢 GSM、主表持平或输 → 改成「极低 bit OPD 稳定性」短文，不要硬写 occupancy。
- #15 hold 后主表仍平 → 先别扩 4B；查数据域、CE=0、还是 ~160 token 轨迹根本没打到长推理占用。

---

## 10. 建议执行顺序（不要并行发明方法）

1. **#15**：8 卡，从 #10 `checkpoint-30` hold `2e-6` 到 100。看 35/50/75/100 的 GSM、jump、五任务。`hold_after_warmup=True`，LR 必须钉住。
2. **#16**：1 空闲卡，BF16 thinking GSM（有余力加 MATH-500）。
3. **#9 / #11**：论文离线主表。没有它们不要宣称方法赢了。
4. 按 #15 结果再改**一处**方法：jump 不稳 → 冻 scale；主表不动 → 加 0.2 teacher-weighted CE 或偏 MATH 的 rollout；GSM 继续涨 → 加长 hold 或补全套评测。
5. #14（4B 离线）和「无 Stage 1」探针可以后做，不挡主表。

不要：再扫 LR；从 #10 step 100 续训；先上 PV-OPD / 4B-OPD；把 reverse KL 改成 forward KL 同时改占用（confound）；只报 GSM 结案。

---

## 11. 想请 Codex 一起判断的问题

1. 主文应坚持 occupancy（OPD vs matched offline），还是先写成 W2 稳定性 + 跳码论文？以 #15/#9/#11 会看到什么为分支。
2. #10 回复只有 ~160 token、trunc=0：这还算「长推理 occupancy」吗？要不要把 rollout/评测预算和论文 8192 的关系写进 limitation？
3. CE=0 的 OPD 是否天生偏向 GSM、抬不动 IFEval/MMLU/GPQA？加 0.2 CE 是补救还是放弃纯 occupancy？
4. 无 Stage 1 的对照放主文还是附录？W3 RTN 0.53% 要不要换成 thinking-on 协议重评？
5. 理论部分保持「引用 Ross/DAgger + 量化放大 \(\varepsilon\) 的上界」，还是削弱成纯经验故事？
6. BitDistiller「student-generated 更差」和我们「在线 student 更好」如何在同一实现里和解（冻住一次的 student 序列 vs 在线更新）？

---

## 12. 关键路径

| 用途 | 路径 |
|--|--|
| 实验记录（只追加，不改旧节） | `OnlineQAT/Exp.md` |
| OPD / 离线训练 | `OnlineQAT/main_e2e_distill.py`，`scripts/run_qwen3_1.7b_opd.sh`，`scripts/run_qwen3_1.7b_reasoningqat.sh` |
| Stage 1 | `OnlineQAT/main_block_qat.py` |
| 评测 | `OnlineQAT/scripts/eval_paper_benchmarks.sh` |
| 旧写作说明 / INT4 稿（不要当当前主表） | `OPT-QAT/arxiv.md`，`OPT-QAT/arxiv/sections/` |
| #10 tag | `Qwen3-1.7B-w2g128-opd-lr2e-6-wu30-ws2e-7-schhold` |

Hold 修复：`main_e2e_distill.py` 的 `scheduler_type_name()` 必须用 enum `.value`。8 卡机要有这版代码再跑 #15。
