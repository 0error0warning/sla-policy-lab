# 伙伴维护预测与个体行动

同一条生命保留原有身体更新。新增预测从 Qwen 的名字词元与原始请求表征估计三种表达获得维护的概率。每次有实际回应且还有下一次行动时，只从选择过的表达学习。Qwen 四个动作词的归一化分数决定行动。八条历史先配对回放，再从相同的八个环境种子分别运行三组自主生命。

[预登记的设计与门槛](PROTOCOL.md) · [结果、轨迹和资源](../../artifacts/model-studies/2026-11-13/partner-prediction/REPORT.md)

## 复核

在仓库根运行，CPU 复算需要 NumPy 和 SciPy。下列命令以 POSIX 路径作示例。原始特征、私有参数、预测、四动作分布和逐轮身体记录都在下载包内。Git 仓库存放固定版本大模型的权重清单。

```sh
study=artifacts/model-studies/2026-11-13
run="$study/partner-prediction/gpu/partnerpred"
mkdir -p /tmp/sla-partner-prediction/input /tmp/sla-partner-prediction/results
python -m zipfile -e "$run/inputs.zip" /tmp/sla-partner-prediction/input
python -m zipfile -e "$run/results.zip" /tmp/sla-partner-prediction/results
python -m experiments.partner_prediction.audit \
  --root /tmp/sla-partner-prediction/results/output \
  --inputs /tmp/sla-partner-prediction/input \
  --source17 "$study/lifetime-cooperation/gpu/lcpilot/results.zip" \
  --source19 "$study/loreft-omission/gpu/omitstable/results.zip" \
  --archive "$run/results.zip" --out /tmp/sla-partner-prediction/analysis
```

在单张支持 BF16 的 NVIDIA GPU 上重跑时，从 inputs.zip 解出的空白工作目录启动。安装包中的 requirements-gpu.txt，将 CUBLAS_WORKSPACE_CONFIG 设为 :4096:8，并运行以下命令。权重版本、文件哈希和运行依赖见输入锁及实际回执。

```sh
python -m experiments.partner_prediction.gpu --root . --mode partnerpred
```

技术复跑复用同一 inputs.zip。将 gpu/recheck/results.zip 解压到另一个目录，再对其单独运行上面的 audit，将结果写到第二个分析目录。下例将两次的原始输出目录与 CPU 分析目录逐一配对。

```sh
python -m experiments.partner_prediction.recheck \
  --first /tmp/sla-partner-prediction/results/output \
  --second /tmp/sla-partner-prediction/recheck/output \
  --analysis /tmp/sla-partner-prediction/analysis \
  --out /tmp/sla-partner-prediction/analysis/recheck
```

policy.csv 给原说法和留出说法在每条生命、每位伙伴、每种预测替换下的动作概率指标。prediction.csv 给预测误差、排序与概率移动。life.csv 保留任务、维护和风险机会的分子分母。immediate.csv 记录每次更新对当前与另一伙伴的即时影响。gates.json 给各项预设门槛的配对差和整生命重抽样区间。
