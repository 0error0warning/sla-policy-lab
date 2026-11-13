# 个体 LoReFT 与 Activation Steering

[前瞻方案](PROTOCOL.md) · [实验报告与原始数据](../../artifacts/model-studies/2026-11-13/loreft-life/REPORT.md)

从同一群体出生权重、同一个显式健康传感和同一张语义动作表开始。训练的经历来自上一轮 Qwen 自主选择的动作与实际后果。本波的 LoReFT 和配对 LoRA 逐步给动作打分并按同一记录更新。原位置 LoRA 训练第 17 轮模型已经具备的身体/伙伴参数，同位置加法和 LoReFT 共享冻结身体通路。LoReFT 两个子空间矩阵与偏置都接受实际后果的梯度。方向由留一环境种子的真实经历差异构造，测试只修改动作决策位置的激活。

## 从归档复核

在仓库根目录运行，CPU 需 NumPy 和 SciPy，重绘图需要 Matplotlib。解包目标应当是新目录。

```sh
ART=artifacts/model-studies/2026-11-13/loreft-life
SRC=artifacts/model-studies/2026-11-13/lifetime-cooperation/gpu/lcpilot/results.zip
python -m experiments.direct_policy.unpack --archive "$ART/gpu/sameresid/results.zip" --receipt "$ART/gpu/sameresid/receipt.json" --destination work/loreft-run
python -m experiments.loreft_life.audit --root work/loreft-run/output --source "$SRC" --out work/loreft-analysis
python -m experiments.loreft_life.figure work/loreft-analysis work/loreft.png
```

重算 Qwen 时，在一个空目录解开归档的 `gpu/sameresid/inputs.zip`，使用支持 BF16 的 NVIDIA GPU 及运行收据所列的软件版本并安装 `requirements-gpu.txt`。worker 校验固定 Qwen 权重、出生文件及来源归档。一次完整运行生成全新的 output 目录。

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m experiments.loreft_life.gpu --root . --mode sameresid
```

每条历史的 `updates.jsonl` 按时序记录六种策略的四动作概率、真实反馈、重要性权重、实际步长及更新前后哈希。`birth.npz`、`at24.npz`、`last-living.npz` 存放每个可训练条件的 FP32 向量。`final-panel.npy` 的轴是参数条件、措辞、健康、上下文、动作。顺序在 `design.ROLES` 与 #17 的 `PANEL` 中定义。每种子的 `steering/probabilities.npy` 多出个人历史（稳定、翻转）和十二种激活操作两个开头轴，之后为措辞、健康、上下文和四个语义动作。供体对比向量、目标的留一单位方向和等范数随机对照均保留。
