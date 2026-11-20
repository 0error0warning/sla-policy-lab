# 实验产物归档（`artifacts/model-studies/`）

按日期保存每轮 GPU 实验的文字总结、审计 JSON、指标表格和绘图。

## 主线报告入口

- **[月度研究报告（2026-10-15 至 2026-11-20）](2026-11-20/monthly-research-report/RESEARCH_REPORT_2026-10-15_to_2026-11-20.md)**：汇总研究动机、11-10 至 11-15 的排查过程，以及 `embodied_open_rl` `v1`–`v4` 的主要定量结论与图表。
- **[`embodied-open-rl-v4` 实验报告](2026-11-18/embodied-open-rl-v4/REPORT.md)**：30 条生命镜像伙伴对照（`r7`）的细项指标。
- 迭代记录：[`v1` (11-17)](2026-11-17/embodied-open-rl/REPORT.md)、[`v2` (11-17)](2026-11-17/embodied-open-rl-v2/REPORT.md)、[`v3` (11-18)](2026-11-18/embodied-open-rl-v3/REPORT.md)。

## 早期排查目录（11-10 至 11-15）

- `2026-11-10/`：冻结模型诊断、身体通道探针与初步社会学习测试（索引见 [`2026-11-10/README.md`](2026-11-10/README.md)，已接入 CI）。
- `2026-11-11/`：闭环验证（`closed-loop/`，已接入 CI）、`direct-policy`、`decision-mechanisms`、`weight-methods`、`baseline-layer3`。
- `2026-11-12/` 至 `2026-11-15/`：围绕损伤权重、致死边界、内部状态读取、LoReFT/激活引导以及自我未来预测的单项拆解（各目录内保留 `REPORT.md` 与 `*.json`）。

## Git 追踪范围说明

为避免拉取仓库时下载几个 GB 的模型权重：

- **纳入 Git 追踪**：全部 `REPORT.md`、协议文本、`audit.json` / `summary.json` / `*.csv` 汇总数据、图表 `.png`，以及 CI 复验所需的打包结果（`2026-11-10/`、`2026-11-11/closed-loop/` 与 `2026-11-18/embodied-open-rl-v4/run/pilot-r7-segments/merged.zip`）。
- **已由 `.gitignore` 忽略**：`*.safetensors`、`*.bin`、`*.pt` 权重文件、`*_chunks/` / `chunks/` / `partial/` 分块目录、单种子 `checkpoints.zip` / `inputs.zip`，以及 `2026-11-11`–`15` 已有文字和 JSON 汇总的中间 `.zip` 归档。
