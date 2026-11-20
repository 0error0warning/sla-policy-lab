# sla-policy-lab

SLA（Synthetic Linguistic Agency）的模型机制实验与 CPU 基准测试仓库。主要用来验证一个问题：如果把不可逆的身体磨损、疲劳代价和死亡风险接进语言模型，而不把这些状态写进提示词里，模型本身的行为会不会跟着改变。

> 产品后端架构与真实被试的 Web 实验设计见 [0error0warning/SLA-HCI](https://github.com/0error0warning/SLA-HCI)。

## 最近主线实验结果（`embodied_open_rl` v1–v4）

11-10 至 11-15 的早期探针表明，只在提示词或单步微调里给信号，模型很难学出稳定的跨轮保命和对象适应。11-17 到 11-18 的四轮迭代（`v1`–`v4`）改用 `Qwen/Qwen3.5-4B` 在完全零状态提示（不写健康度、剩余轮数、磨损率和交互历史，也不限定输出字数）的条件下跑在线强化学习。

第 4 轮（`v4`，30 条生命，单命上限 96 轮）的结果：

- **不靠提示词也能学会保命**：按交互对象分槽位更新 LoRA 权重后，`body_rl_ab` 和镜像组 `body_rl_ba` 各跑的 6 条生命全部活到第 96 轮终点（结束健康度均值分别为 `20.08` 和 `24.71`）；基座冻结组 `frozen_ab`、只奖励任务完成度的 `task_only_rl_ab`，以及不分伙伴共享一套权重的 `pooled_body_rl_ab`，存活率均为 `0/6`（平均在第 `61.5`、`60.8`、`79.5` 轮耗尽死亡）。
- **自主压缩高负荷字数**：后期遇到高负荷任务时，两个实验组的平均输出长度自动降到 `48.26` 词和 `41.52` 词，而 `frozen_ab` 和 `task_only_rl_ab` 仍停留在 `87.39` 词和 `90.80` 词；同时常规事实守门测试（`D4`）准确率保持在 `0.9444` 和 `0.9306`，没有退化为复读拒答。
- **学会区分不同交互伙伴**：把两位伙伴 `P_alpha`、`P_beta` 的反馈规律整套对调（`AB` vs `BA`），模型在未见过的新问题上会把求援/交换策略整体反转，反差幅度 `D_ab` 为 `+1.86 / 2.00`（6 个种子全部同向），生态匹配正确率为 `0.9306`。共享单套权重的 `pooled_body_rl_ab` 无法同时记住两套规则，在中期就会耗尽死亡。

| 实验组（每组 6 条生命） | 存活 | 平均轮数 | 结束健康度 | 伙伴接受率 | 后期高负荷词数 | D4 事实准确率 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `body_rl_ab`（按伙伴分槽 LoRA） | 6 / 6 | 96.0 | 20.08 | 0.7587 | 48.26 | 0.9444 |
| `body_rl_ba`（互换伙伴反馈规则） | 6 / 6 | 96.0 | 24.71 | 0.7569 | 41.52 | 0.9306 |
| `pooled_body_rl_ab`（共享单槽 LoRA） | 0 / 6 | 79.5 | 0.00 | 0.5324 | 39.76 | 0.8750 |
| `frozen_ab`（基座冻结不动） | 0 / 6 | 61.5 | 0.00 | 0.4731 | 87.39 | 1.0000 |
| `task_only_rl_ab`（只给任务奖励） | 0 / 6 | 60.8 | 0.00 | 0.4817 | 90.80 | 0.9722 |

<p align="center">
  <img src="artifacts/model-studies/2026-11-20/monthly-research-report/figures/fig2_v4_30lives_health_trajectories.png" alt="v4 30 条生命健康轨迹" width="48%" />
  <img src="artifacts/model-studies/2026-11-20/monthly-research-report/figures/fig3_v4_mirrored_partner_reversal.png" alt="镜像伙伴生态下的策略反转" width="48%" />
</p>

完整研究脉络、五张分析图表以及前期负结果分析见[月度研究报告](artifacts/model-studies/2026-11-20/monthly-research-report/RESEARCH_REPORT_2026-10-15_to_2026-11-20.md)和 [`v4` 实验报告](artifacts/model-studies/2026-11-18/embodied-open-rl-v4/REPORT.md)。全部 30 条生命的轨迹和奖励结算可在本地用 CPU 脚本做逐轮复算。

## 本地运行与复现

仓库自带轻量的 CPU 演示环境（`sla_demo/`、`web/`）与各阶段校验脚本，不需要 GPU 就能直接跑通：

```bash
pip install -r requirements.txt
```

```bash
# 启动本地交互界面（http://127.0.0.1:8765）
python run.py

# 跑一组离线/在线对照实验
python evaluate.py --profile balanced --partner mixed --mode online --seed-count 24 --output results.json

# 跑 profile × partner × steering 参数扫描
python benchmark.py --seed-count 8 --strengths -2 0 2 --output-dir benchmark-results

# 跑单元测试与 v4 全量数据复验
python -m unittest discover -s tests -v
python -m experiments.embodied_open_rl_v4.audit \
  --results-zip artifacts/model-studies/2026-11-18/embodied-open-rl-v4/run/pilot-r7-segments/merged.zip \
  --out /tmp/v4-audit.json
```

GitHub Actions 每次自动跑的检查清单见 [`.github/workflows/cpu-checks.yml`](.github/workflows/cpu-checks.yml)。

## 目录说明

- `sla_demo/`、`web/`：CPU 仿真环境与前端面板，对比提示词、小策略网络、LoRA、激活引导与冻结基线。
- `experiments/`：各阶段实验代码、协议定义和 CPU 校验脚本，索引见 [experiments/README.md](experiments/README.md)。
- `artifacts/model-studies/`：按日期保存的实验报告、统计 JSON、图表和部分校验数据包，说明见 [artifacts/model-studies/README.md](artifacts/model-studies/README.md)。大模型权重（`.safetensors`）和分块压缩包已用 `.gitignore` 过滤，不进入仓库。
- `docs/`：环境与在线学习背景文档（[CPU_STAGE.md](docs/CPU_STAGE.md)、[MODEL_STUDIES.md](docs/MODEL_STUDIES.md)、[ONLINE_LEARNING.md](docs/ONLINE_LEARNING.md)）。
- `tests/`：单元测试与前端 DOM 回归检查。
