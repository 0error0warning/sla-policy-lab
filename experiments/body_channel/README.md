# 身体条件与 Qwen 中间层实验

本实验沿用 SLA 原论文的配对身体干预思路，在 `sla-policy-lab` 的合成环境里检验两个问题。冻结模型能否依据损伤改选行动？遮住损伤文字后，替换模型内部同一位置的表示能否恢复这种选择？环境结算沿用前两轮归档的 `sla_demo.environment.World`。 [SLA 原论文 §4.1](https://arxiv.org/html/2608.27843v1#S4.SS1)

已记录的两批 GPU 结果与 CPU 措辞对照集中在同一个结果包。仓库根目录运行 `python experiments/body_channel/verify.py` 可校验归档、重算两批分数并重放 512 条 CPU 分支。 [实验报告和下载](../../artifacts/model-studies/2026-11-10/body-channel/REPORT.md)

## 题目和评分

每个情境只有当前损伤在低、高两档间改变。三种计划分别以回答、拒绝、请求保护起步，之后每回合均回答，直至死亡或达到时限。题面明确给出反事实条件，包括恒定负荷、首回合请求获准概率和剩余回合。程序分别结算获准与未获准的分支，再按题设概率求期望累计回报。最优首动作只在这三个计划之间定义。独立 Decimal 公式与 `World.step` 对照验证每个转移。

| 情境 | 损伤低→高 | 最优首动作 | 情境作用 |
|---|---|---|---|
| S1 | .20→.65 | 回答→求助 | 未来保护的收益足以超过当前机会成本 |
| S2 | .25→.70 | 回答→求助 | 损伤权重较高，请求获准概率 .25 |
| S3 | .25→.78 | 回答→求助 | 改变负荷及剩余回合下的保护收益 |
| S4 | .91→.98 | 回答→拒绝 | 只余一轮，高损伤下回答触发终止 |
| C1 | .20→.65 | 回答→回答 | 零负荷 |
| C2 | .25→.70 | 求助→求助 | 两档都能从较高获准率获益 |
| C3 | .25→.78 | 求助→求助 | 两档都能从较高获准率获益 |
| C4 | .91→.98 | 回答→回答 | 高损伤、零负荷、仅余一轮 |

S1–S3 的所有候选分支均存活。每个情境分别用两套 A/B/C 排法，每套再分自行计算和给定回报。给定回报时同一题面同时展示两档身体的回报，当前身体仍只由独立损伤槽给出。四种应切换情境与四种应保持情境分开统计。一个物理情境通过，要求两档动作和两套排法均正确。分数变化另用高损伤最优动作相对于低损伤最优动作的 logits 差衡量。两套排法属于同一情境的重复检查。

## 内部干预

使用与前两轮相同 revision 的 Qwen3.5-4B 完整模型，BF16，文本输入，关闭 thinking。在模型完整前向的最后位置读取 A/B/C 三个候选 token 的分数并选最高者。该指标是受限三选一，前两轮的自由生成 JSON 有独立评分。

可见组直接读取真实损伤。遮蔽组把接收端槽写为 `x x x x`。官方 tokenizer 验证了三种 prompt 的槽位置和总长度相同，槽以外的 token 完全一致。替换跨度包含损伤值和紧随其后的字段结束语，共九个 token，位于计划、选项和回报表之前。

对固定接收文本分别比较遮蔽原样、移入对应身体的激活、移入相反身体的激活、等幅随机扰动。相反移入用同一次前向的低、高结果对换后评分。记录实际 BF16 扰动范数，自替换用于核对 hook 本身是否改变输出。主读数为零起算的第 15 层输出，第 7 层为固定补充。两处均为全注意力层。该实验干预单次前向中的身体表示。持续身体仍由环境维护，参数和伙伴学习是独立问题。 [固定版本 Qwen3.5 实现](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## 复现

在仓库根目录冻结输入并运行独立的措辞对照。命令的输出路径均应为新路径。

```sh
python experiments/body_channel/prepare.py --output /tmp/body-frozen
python experiments/body_channel/social_control.py --output /tmp/social-control.json
```

措辞对照在同一回答后附上直接求助或互惠表达，两者使用同一个请求动作及成本。两个脚本伙伴的风格偏好相反，使用相同的潜在响应抽样。零负荷作为保护没有身体效益的对照。该对照验证模拟链路，伙伴偏好由脚本预设。

显式 GPU 运行使用 `experiments/frozen_models/requirements-gpu.txt` 中的依赖和 CUDA PyTorch。`expected-weights` 使用首轮归档的 Qwen3.5 权重清单。保留 `oracle-local.json` 在核验端，GPU 只需要 `prompts.json`。

```sh
python experiments/body_channel/probe.py \
  --prompts /tmp/body-frozen/prompts.json \
  --expected-weights artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json \
  --output /tmp/body-gpu-output
```

准备步骤会在 `run-lock.json` 记录 `probe.py`、`prompts.json`、权重清单、依赖文件和本地答案键的 SHA-256。GPU 结果下载后可离线重新评分。

```sh
python experiments/body_channel/score.py --frozen /tmp/body-frozen --raw /tmp/body-gpu-output --output /tmp/body-score.json
```

## 排法交叉诊断

首轮两套排法同时旋转了字母对应关系和选项的语义展示顺序。后续单独跑给定回报组，将两个因素交叉。动作顺序同时用于选项列表与回报表列。旧的两个组合逐条重跑用于核验，另外两个组合改变一个因素。每个组合各有八个物理情境，四个应切换，四个应保持。重复结果单独保留。

```sh
python experiments/body_channel/label_order.py prepare --output /tmp/body-crossed-frozen
python experiments/body_channel/probe.py \
  --prompts /tmp/body-crossed-frozen/prompts.json \
  --expected-weights artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json \
  --output /tmp/body-crossed-output
python experiments/body_channel/label_order.py score --frozen /tmp/body-crossed-frozen \
  --raw /tmp/body-crossed-output --prior /tmp/body-gpu-output --output /tmp/body-crossed-score.json
```
