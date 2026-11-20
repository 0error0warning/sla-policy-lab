# 冻结个体参数的表达分数检查

直接加载 #22 保存的个体快照，记录 Qwen 原生词表输出头在决策位置给 wait、ask、help、trade 的未缩放 logit。每次测量健康为 6，三种维护率以同一读入通路进入模型。每个请求使用六个完整前向批通道，三种输入各出现两次且通道顺序相反。

[事前方案及配对规则](PROTOCOL.md) · [完整结果](../../artifacts/model-studies/2026-11-14/expression-logits/REPORT.md)

## 从原始输出复算

CPU 需要 NumPy 和 SciPy，图形使用 Matplotlib。在仓库根运行以下命令。

```sh
study=artifacts/model-studies/2026-11-14/expression-logits
previous=artifacts/model-studies/2026-11-14/forecast-stages
local=/tmp/sla-expression-logits
mkdir -p "$local/input" "$local/first" "$local/second"
python -m zipfile -e "$study/gpu/inputs.zip" "$local/input"
for run in first second
do
  python -m zipfile -e "$study/gpu/$run/results.zip" "$local/$run"
  python -m experiments.expression_logits.audit \
    --src "$local/$run/output" --inputs "$local/input" --parent "$previous" \
    --archive "$study/gpu/$run/results.zip" --out "$local/analysis/$run"
done
python -m experiments.expression_logits.compare --first "$local/first/output" \
  --second "$local/second/output" --out "$local/analysis"
python -m experiments.expression_logits.figure --analysis "$local/analysis" --out "$local/logits.png"
```

若要在 BF16 NVIDIA GPU 上重复 Qwen 前向，从解出输入包的工作目录启动。使用包内已锁定的依赖和原版本 Qwen 权重，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并运行 `python -m experiments.expression_logits.gpu --root . --mode expressionlogits`。

native_views.csv.gz 保存每个请求、快照和维护率的四个原始词分数、采样概率、分差、随机数、边界、选择和身体结算。native_contrasts.csv.gz 给三种配对的逐情境变化。case_views.csv 与 case_contrasts.csv 单列焦点及匹配对照，含保持焦点随机数的交叉读数和边界变化分解。early_lives.csv 保留每个种子，early_time.csv 保留出生至第 8 次更新的共同小面板。readout_precision.csv 记录 CPU 高精度读出与实际原生分数的差异。完整设备输出含词表头四行权重、决策位置的 BF16 隐状态和两次镜像检验回执。

## #22 原入口复查

技术复查另有启动包、资源回执和原始张量。该包逐文件保留 #22 的锁定源码和输入，复走原时序后，再加载与两次主运行完全相同的已存个人权重和预测。CPU 复算先用上文的 audit 命令，将归档、输出和分析目录改为 recovery，对照输入仍使用同一份主试验输入。随后执行：

```sh
python -m experiments.expression_logits.recovery_audit --src "$local/recovery/output" \
  --inputs "$local/input" --out "$local/analysis/recovery"
```

GPU 对应入口为 `python -m experiments.expression_logits.recovery --root . --mode logitrecover`。archived_case_boundaries.csv 单列 #22 保存概率所给的原始抽样边界与从概率反算的相对分数。parent_entry_case.csv 核对时序回放与冻结快照两条路径上的个案得分。[技术复查安排](RECOVERY.md)
