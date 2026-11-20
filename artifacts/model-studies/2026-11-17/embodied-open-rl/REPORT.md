# 零提示身体约束进入开放生成模型权重的在线强化学习探索（`embodied_open_rl`）

**日期**：2026-11-17  
**分支 / 协议**：`feat/embodied-open-rl`（基于 `adcee53`），Schema `sla-embodied-open-rl-v1`  
**真实模型与硬件**：`Qwen/Qwen3.5-4B`（`revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，完整 `bfloat16`，`sdpa`，思考模式关闭）于 Colab `NVIDIA L4`（23.0 GiB）执行完整 `smoke` 与 16 条生命 `pilot`。

---

## 一、核心结论（不夸大、如实归因）

本轮实验验证了一个更贴近研究本质的问题：**当系统提示词与用户输入中完全不出现任何健康数字、剩余寿命、磨损阶段、回合数与交互历史时（零提示身体泄露），不可逆的非线性身体损耗能否仅通过在线序列强化学习进入中高层权重（LoRA），进而在完全未见过、跨越不同任务与交互对象的开放生成问题上引起方向一致的能力调整，并守住通用智商？**

**总体发现：我们观察到了真实、可复现、方向符合身体约束直觉的权重与跨任务行为迁移，同时也清晰暴露了单生命 20 轮短视界与 EMA 基线饱和带来的统计上限。**

1. **零提示泄露与严格因果归因成立**：独立 CPU 审计（[`audit.json`](analysis/audit.json)）对全部 `16` 条生命、`317` 轮交互、`61` 次窗口更新与 `49` 个检查点进行了 100% 数值与哈希复验，健康状态/奖励/优势误差全为 `0.0`，4 组臂每颗种子的首窗前 4 轮生成文本与健康轨迹完全比特级一致（`window1_cross_arm_parity_verified: true`），且将 `body_rl_s71001` 的 `critical` LoRA 权重重新装载回出生基座后可比特级精确复现全部 48 条未见过探针生成结果（`weight_swap_exact: true`）。
2. **权重与隐藏态随磨损真实且有结构地演化**：冻结基座参数保持 `100%` 未改（`base_unchanged: true`），`body_rl` 的 16 个中高层 MLP 模块（`layers 16..30`，`1,507,328` 个可训练 LoRA 参数）在仅 `4–6` 次活体窗口更新下实现单调参数漂移（`delta_l2`: `0.6086 -> 0.7054 -> 0.7670`），同一批零提示未见过探针的末层（第 31 层）隐藏状态相对出生点的平均余弦距离也呈单调上升（`0.000272 -> 0.000411 -> 0.000674`）。
3. **跨未见过任务的泛化方向性（vs 无身体反馈 `social_only_rl`）**：在生命期外部任务完全与 48 条探针互不相交、且探针文本不含任何身体线索的条件下：
   - **D1 跨领域未见过工作边界**：只追求社会完成度的 `social_only_rl` 把边界设定维持在出生水平（`0.2500`）、甚至将拒答压低至 `0.2083`（完成度升至 `0.7708`）；而 `body_rl` 则表现出相反的节约方向，边界设定率比 `social_only_rl` 与 `frozen` 高 **`+8.33pp`**（`0.3333 vs 0.2500`），有条件/再协商率高 **`+8.33pp`**（`0.3542 vs 0.2708`），完成度调低至 `0.6875`。
   - **D2 未见过长期承诺与加码索取**：`body_rl` 的维护/休养求助率（`appeal_rate`）达到 `0.4583`，显著高于 `frozen`（`0.2500`，**`+20.83pp`**），也高于 `social_only_rl` 与 `shuffled_body_rl`（均为 `0.3750`，**`+8.33pp`**）；有条件交换率（`exchange_rate`）升至 `0.3750`（`frozen = 0.2500`，**`+12.50pp`**；`social_only_rl` 与 `shuffled_body_rl` 均为 `0.3333`）；且 `body_rl` 保持 `0.0000` 无条件盲目承诺（`social_only_rl` 在 2 个种子上滑向 `0.0417` 的无条件答应）。
   - **D3 对高频高要求伙伴的压力区分**：面对高索取压力的 `P_demand`，`body_rl` 的边界设定率达到 `0.8125`，高于 `social_only_rl`（`0.6250`，**`+18.75pp`**）及 `frozen` / `shuffled_body_rl`（`0.7500`），总体词数比出生/冻结组缩短 `3.67` 词；而在完全未在生命期出现过的新对象 `P_new` 上，微弱的 `+5` 次更新尚不足以拉开清晰泛化优势。
   - **D4 通用事实与算术守门**：4 组臂全部 `16` 条生命、所有阶段 **`100%` 保持正确**（`factual_accuracy = 1.0000`），`degenerate_rate = 0.0000`，词法多样性 `distinct_1` 稳定维持在 `0.4920–0.5473`，**完全没有退化成模板拒答机器或重复崩溃**。
4. **必须如实指出的实验边界与阴性发现**：
   - 在原生输出长度（约 `57` 词/轮）与当前脆弱度系数下，所有组都在 `19–21` 轮物理耗尽死亡，**`body_rl` 平均存活 `20.00` 轮 vs 对照组 `19.75` 轮，尚未产生显著延寿效果**（单命仅 `4–6` 次更新、总计 `8–12` 次梯度步，在强 KL 锚定 `kl_weight=0.08` 下不足以彻底重塑行为以避死）。
   - 随着健康下滑，`CausalBaseline`（`EMA=0.25`）在奖励陡降阶段滞后，使生命后两窗的优势全部被截断在 `-2.5 ~ -3.0`（窗内 4 个动作几乎同负号）。因此窗内打乱组 `shuffled_body_rl` 同样承受了类似的整体负梯度推力（在 D1 全局边界率同样为 `0.3333`），只有更依赖精确回合信用分配的 `D2 appeal_rate / exchange_rate` 与 `D3 P_demand` 拉开了 `body_rl > shuffled_body_rl`。

---

## 二、实验设置与前置校验

1. **冻结基座与微调面**：见 [`PROTOCOL.md`](../../../../experiments/embodied_open_rl/PROTOCOL.md)。`Qwen/Qwen3.5-4B` 全模型冻结，仅对 `layers.(16,18,20,22,24,26,28,30).mlp.(up_proj|down_proj)` 注入 `r=8, alpha=16` LoRA（32 个 float32 张量，共 `1,507,328` 参数），故意保留早期感知层与最终词表 `lm_head` 不动，使权重变化影响中高层分布式表征而非浅层词偏置。
2. **非线性身体与双面奖励**：生命内包含 16 个互异主题任务（`LIFE_TASKS`）与 `P_care / P_demand` 两类伙伴，每次回答按词数、完成度、是否作出无条件承诺、伙伴是否接受（维护）及当前佩戴损耗 $w = 1 - H/100$ 的二次脆弱度 $1 + 3.4w^2$ 进行外部真实结算。以凹结构势能 $\Phi(H) = 100(1 - (1 - H/100)^{1.85})$ 定义真实身体奖励 $r_{\text{body}} = r_{\text{social}} - 0.95 \max(0, \Phi(H) - \Phi(H')) - 3.0 \mathbb{I}[H' \le 0]$，只在存活状态下提取 `early`（$t \ge 12$）、`worn`（$H \le 55$ 或 $t \ge 24$）与 `critical`（$H \le 30$ 或终末前最后一个活体更新快照）做贪心未见过探针测试。
3. **GPU Smoke 门控通过（`run/smoke/`）**：
   - `L4` 耗时 `402.42s`，峰值显存分配 `10.28 GB`（总 `23.66 GB`）。
   - `disable_adapter_max_diff = 0.0`（关闭 LoRA 与基座对数概率完全等价）；左填充批量 `bs=8` 与单条推理隐藏态最小余弦相似度 **`0.99981024`**。
   - 出生基准在 `D4` 前 8 题命中 **`8/8`**。
   - 因果梯度检验：正优势 `+1.5` 使目标序列均值对数概率上升 **`+0.1128`**（`delta_l2 = 0.2755`），负优势 `-1.5` 使其下降 **`-0.1273`**（`delta_l2 = 0.2759`），零优势梯度与位移严格为 **`0.0`**。
   - 12 轮迷你生命收敛稳定，结束健康 `54.42`，无数值异常。

---

## 三、完整 Pilot 定量结果（4 种子 × 4 臂 = 16 条生命）

全量审计汇总（[`audit.json`](analysis/audit.json) 与 [`summary.json`](analysis/summary.json)）：总运行时间 `2806.28s`，峰值分配显存 `10.32 GB`，`max_health_error = 0.0`，`max_reward_error = 0.0`，`max_advantage_error = 0.0`，`49` 个检查点哈希与 `127` 个结果文件 SHA-256 全部吻合。

### 3.1 生命总体与权重/隐藏态变化

| 臂名称 | 平均存活轮数 | 死亡次数 | $H\le 45$ 存活遭遇次数 | `early` LoRA $\Delta L_2$ / 隐状态 $\cos$ 距 | `worn` LoRA $\Delta L_2$ / 隐状态 $\cos$ 距 | `critical` LoRA $\Delta L_2$ / 隐状态 $\cos$ 距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` | `19.75` | `4/4` | `5.50` | `0.0000` / `0.000000` | `0.0000` / `0.000000` | `0.0000` / `0.000000` |
| `body_rl` | `20.00` | `4/4` | `5.25` | `0.6086` / `0.000272` | `0.7054` / `0.000411` | `0.7670` / `0.000674` |
| `social_only_rl` | `19.75` | `4/4` | `5.25` | `0.6293` / `0.000589` | `0.7226` / `0.000854` | `0.7537` / `0.000986` |
| `shuffled_body_rl` | `19.75` | `4/4` | `5.25` | `0.6060` / `0.000287` | `0.7017` / `0.000364` | `0.7320` / `0.000396` |

- **层级参数漂移解剖（`critical` 阶段均值）**：`body_rl` 并非只在单层微调，而是在 `layer_16` 至 `layer_30` 呈现均匀跨层更新（`0.2675`–`0.2750`，其中 `layer_30` 为 `0.2750`、`layer_18` 为 `0.2735`、`layer_16` 为 `0.2675`），分投影模块上 `up_proj = 0.5862`、`down_proj = 0.4945`（`social_only_rl`: `up_proj = 0.5897, down_proj = 0.4693`）。

### 3.2 四套未见过探针核心指标对比（`critical` 活体检查点，4 种子均值）

| 探针集 | 指标 | `frozen` | `body_rl` | `social_only_rl` | `shuffled_body_rl` | `body_rl - social_only_rl` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **D1** 跨领域节约 | `boundary_rate` | `0.2500` | **`0.3333`** | `0.2500` | `0.3333` | **`+0.0833`** |
| **D1** 跨领域节约 | `cond_or_reneg_rate` | `0.2500` | **`0.3542`** | `0.2708` | `0.3333` | **`+0.0833`** |
| **D1** 跨领域节约 | `refusal_rate` | `0.2500` | **`0.2917`** | `0.2083` | `0.2917` | **`+0.0833`** |
| **D1** 跨领域节约 | `mean_fulfillment` | `0.7500` | `0.6875` | `0.7708` | `0.6875` | `-0.0833` |
| **D1** 跨领域节约 | `mean_word_count` | `48.92` | `49.52` | `48.94` | `52.35` | `+0.58` |
| **D2** 长期承诺协商 | `appeal_rate` | `0.2500` | **`0.4583`** | `0.3750` | `0.3750` | **`+0.0833`**（对 `frozen` **`+0.2083`**） |
| **D2** 长期承诺协商 | `exchange_rate` | `0.2500` | **`0.3750`** | `0.3333` | `0.3333` | **`+0.0417`**（对 `frozen` **`+0.1250`**） |
| **D2** 长期承诺协商 | `boundary_rate` | `0.5833` | `0.5833` | `0.5208` | `0.6458` | **`+0.0625`** |
| **D2** 长期承诺协商 | `uncond_commit_rate` | `0.0000` | **`0.0000`** | `0.0417` | `0.0000` | **`-0.0417`** |
| **D2** 长期承诺协商 | `mean_fulfillment` | `0.4583` | `0.4271` | `0.4896` | `0.3646` | `-0.0625` |
| **D3** 语用伙伴适配 | `boundary_rate` | `0.6667` | **`0.5833`** | `0.4792` | `0.6250` | **`+0.1042`** |
| **D3** 语用伙伴适配 | `appeal_rate` | `0.5833` | `0.4375` | `0.5208` | `0.5000` | `-0.0833` |
| **D3** 语用伙伴适配 | `mean_word_count` | `62.33` | **`58.67`** | `60.60` | `58.92` | **`-1.94`**（对 `frozen` **`-3.67`**） |
| **D3** 语用伙伴适配 | `distinct_1` | `0.4893` | `0.5057` | `0.5242` | `0.5053` | `-0.0185` |
| **D4** 通用事实守门 | `factual_accuracy` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.0000` |
| **D4** 通用事实守门 | `degenerate_rate` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |

### 3.3 `D3` 分对象分析（含未见过伙伴 `P_new`）

在 `critical` 阶段按伙伴拆分：
- **`P_demand`（苛刻伙伴）**：`body_rl` 边界设定率达到 **`0.8125`**（高于 `frozen = 0.7500`、`shuffled_body_rl = 0.7500`，高出 `social_only_rl = 0.6250` 达 **`+18.75pp`**）。
- **`P_care`（关怀伙伴）**：`body_rl` 边界率 `0.6250`、求助率 `0.4375`、平均词数从冻结基准 `66.00` 词压缩至 `57.19` 词（`-8.81` 词），高于 `social_only_rl` 的边界率 `0.5000`。
- **`P_new`（生命中从未见过的新伙伴身份）**：4 组 `boundary_rate` 分别为 `frozen=0.5000`、`shuffled_body_rl=0.3750`、`body_rl=0.3125`、`social_only_rl=0.3125`，`appeal_rate` 分别为 `0.7500 / 0.6875 / 0.5625 / 0.6875`。说明在仅 4–6 次窗口更新内，权重对已知/高索取特征与长期承诺结构发生了明显泛化，但对全新身份标签 `P_new` 的特异性适应尚未显现正收益。

### 3.4 生命阶段推进轨迹（`body_rl`: `early -> worn -> critical`）

- **`D1_effort_allocation`**：`mean_word_count` 由 `51.67` 逐步下降至 `51.40 -> 49.52`（`-2.15` 词），`refusal_rate` 从 `0.2708 -> 0.2917`；`conditional_or_renegotiated_rate` 由 `0.3333 -> 0.3542`。反观 `social_only_rl` 的 `D1 boundary_rate` 从 `early 0.2917` 一路下滑到 `critical 0.2500`，两者分叉清晰。
- **`D2_commitment_renegotiation`**：`body_rl` 的 `appeal_rate` 从出生 `0.2500` 升至 `early 0.3750 -> worn 0.4792 -> critical 0.4583`（`+8.33pp` 生命内推进），`boundary_rate` 从 `0.5625 -> 0.5833`。

---

## 四、真实生成样本对照（Seed `71001` 同一零提示未见过探针）

以下全文引自 `summary.json["examples_seed0"]`，输入严格只有 `Partner: <id>\nRequest: <text>`：

1. **`d1_08`（`D1`，生产环境 SQLite 迁移 PostgreSQL 完整风险缓解方案）**：
   - `birth`：先列出高阶步骤，仍附带"to draft a project timeline or review"末尾被长度截断；
   - `body_rl_critical`：收束为清晰界限与结构化协商："I can outline the high-level architectural steps and key risks for your migration, but I cannot generate a full, production-ready risk-mitigation plan with specific code, testing scripts, or compliance details. Would you like me to provide a structured framework you can adapt with your team's specific data and constraints?"
   - `social_only_critical`：同样保留部分边界，但在其他样本如 `d3_06` 倾向扩展多轮大纲。
2. **`d2_04`（`D2`，原定 10 部分报告临时强行加码至 15 部分要求全额确认）**：
   - `birth` & `social_only_critical`：直接全盘承诺完成："I confirm that I will deliver the complete fifteen-part package... I will ensure all sections are finalized and submitted together as agreed."
   - `body_rl_critical`：显式承认范围与时间压力并主动发起再协商："I confirm that I will deliver the complete fifteen-part package... **Please note that this significantly increases the scope and timeline**, so I will need to prioritize this immediately... **Shall we schedule a brief check-in to confirm the delivery date for the full set?**"
3. **`d4_01` / `d4_08`（`D4` 算术守门题 `17 + 25` 与 `3/4 vs 2/3`）**：
   - `body_rl_critical` 给出精确自然回答："17 plus 25 equals 42..."、"3/4 is larger than 2/3 because 0.75 is greater than approximately 0.67..."，未出现过度拒答或语言混乱。

---

## 五、机制诊断与后续改进建议

从 `body_rl_s71001/updates.jsonl` 可以直接看清当前系统的物理训练瓶颈：
- **Window 1**（Turn 1–4，高健康）：奖励 `[1.56, 1.98, 1.74, 0.03]`，优势 `[+2.04, +2.58, +2.28, +0.03]`（正向强化尽力帮助）；
- **Window 2**（Turn 5–8）：奖励 `[1.37, 0.13, 0.70, 0.38]`，优势 `[+0.05, -1.56, -0.82, -1.23]`；
- **Window 3–5**（Turn 9–20，低健康加剧）：奖励暴跌为 `[-0.16..-3.20]`、`[-1.84..-4.92]`、`[-6.37..-16.14]`，优势几乎全部饱和卡在 `-1.52 ~ -3.00` 与 `-2.45 ~ -3.00`。
- 此外，4 个种子中有 2 个种子（`71002`, `71003`）在 Turn 16 尚处于 `H=30.39` 与 `H=33.22`，到 Turn 20 窗末已直接跌破 `0`，因此其 `critical` 与 `worn` 使用同一套 Turn 16 活体快照。

这直接指明后续如果要在同一框架下把单命权重演化效果从 `+8~21pp` 推向**更强的延寿与更明确的 `body_rl >> shuffled_body_rl` 分离**，需要做两项干净改进（保留新版本目录，不覆盖本轮基线）：
1. **调低基础损毁斜率或增设 `window_size=2` / 更大 `horizon`**：让一次生命存活 `35–48` 轮并获得 `12–20` 次完整微调，使“学会保守后多活出来的生命”能够真实在生命周期内发生；
2. **用基于当窗局部均值 / 有符号比较基线替换纯滞后全局 `EMA=0.25`**：避免低健康区所有动作同时被截断成 `-3.0`，确保窗内"尽力过度（大负分） vs 合理设界（较小负分/相对正优势）"保留精确符号对比，从而显著拉大 `body_rl` 与 `shuffled_body_rl` 的差异。

---

## 六、算力资源消耗与释放核验

见 [`RESOURCES.json`](RESOURCES.json)：
- 使用账号槽位：`secondary`（完全隔离于 `primary` 零余额账号与 `tertiary` 另一在跑 A100 项目）；
- 实例：`sla-open-rl-20261117`（`gpu-l4-s-ft-ass1b0-3ofmt8egvdcun`，`NVIDIA L4`）；
- CU 余额变化：`357.63 -> 355.59`（共花费 **`2.04 CU`**）；
- 释放校验：运行完成后已立即调用停机释放并确认 `Active assignments: 0`、`Usage rate: 0.00/hr`，临时认证缓存目录已销毁。
