# 实验模块索引（`experiments/`）

这里汇总 2026-11-10 至 2026-11-18 期间的所有实验代码包。目录名保持与旧命令兼容，方便继续通过 `python -m experiments.<目录名>...` 调用或运行 `tests/`。

## 1. 主线实验：零状态提示的具身在线强化学习（11-17 至 11-18）

验证当提示词里不出现任何健康数值、剩余轮数和历史提示时，`Qwen/Qwen3.5-4B` 能否靠真实身体结果学会省力保命与区分对象。

| 模块 | 日期 | 内容 | 对应报告 |
| --- | --- | --- | --- |
| `embodied_open_rl` (`v1`) | 11-17 | 首版 16 条生命零提示在线 RL，对照 `social_only_rl`、`shuffled_body_rl`、`frozen` | [`v1` 报告](../artifacts/model-studies/2026-11-17/embodied-open-rl/REPORT.md) |
| `embodied_open_rl_v2` (`v2`) | 11-17 | 扩展至 24 条生命、引入字数磨损机制与半衰期对照 | [`v2` 报告](../artifacts/model-studies/2026-11-17/embodied-open-rl-v2/REPORT.md) |
| `embodied_open_rl_v3` (`v3`) | 11-18 | 24 条生命三阶段（消耗—修复—再挑战），新增 `no_heal_body_rl` 对照 | [`v3` 报告](../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/REPORT.md) |
| `embodied_open_rl_v4` (`v4`) | 11-18 | **30 条生命镜像伙伴生态（`r7`）**：按伙伴分槽 LoRA vs. 共享单槽 vs. 只算任务奖励 | [`v4` 报告](../artifacts/model-studies/2026-11-18/embodied-open-rl-v4/REPORT.md) |

## 2. 基础验证与闭环探针（11-10 至 11-11）

已在 `.github/workflows/cpu-checks.yml` 中接入自动核验的基础实验：

- `frozen_models`：测冻结小模型在状态转移和延时规划任务上的表现（`verify.py`）。
- `body_channel`：检测模型激活与生成是否能感知健康状态差异（`verify.py`）。
- `social_learning`、`social_closed_loop`：在闭环社会反馈下对比 LoRA 与激活引导效果（`verify.py`、`audit.py`、`fit_verify.py`）。

## 3. 早期机制拆解实验（11-11 至 11-15）

在收敛到 `embodied_open_rl` 主线前做的单项诊断与对照实验：

- **直接策略与后果拟合**：`direct_policy`、`consequence_policy`、`consequence_warmstart`
- **损伤惩罚、致死边界与健康基线**：`damage_penalty`、`conditional_penalty`、`health_baseline`、`first_fatal`、`lethal_boundary`
- **单生命轨迹、记忆写入与子空间位置**：`individual_life`、`internal_body`、`lifetime_cooperation`、`loreft_life`、`loreft_omission`
- **对象泛化、改写迁移与语序对照**：`order_transfer`、`paraphrase_transfer`、`snapshot_transfer`、`partner_damage`、`partner_prediction`
- **自我预测、对数概率与决策解耦**：`expression_logits`、`forecast_stages`、`frozen_capability`、`frozen_interface`、`transparent_body`、`body_prediction`、`self_future`
