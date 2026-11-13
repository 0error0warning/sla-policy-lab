# 生命早中末的维护率干预

沿用八条来源历史与 #21 的同一出生参数、身体反馈和监督更新。每次实际决策之前保存完整策略和预测头。固定这些参数、请求和健康后，仅替换提供给模型的维护概率。原任务与留出说法一一配对，健康分别取 6、90、300。重复使用同一随机数得到的动作变换，与使用全部随机数区间积分的预计动作变换分开存储。

[事前指标和时间点](PROTOCOL.md) · [结果与逐生命证据](../../artifacts/model-studies/2026-11-14/forecast-stages/REPORT.md)

## 复核两次运行

以下命令从仓库根执行。CPU 复核需要 NumPy 和 SciPy。图形还用到 Matplotlib。研究归档目录使用本机台北日期。

```sh
prev=artifacts/model-studies/2026-11-13/partner-prediction
study=artifacts/model-studies/2026-11-14/forecast-stages
dest=/tmp/sla-forecast-stages
mkdir -p "$dest/input" "$dest/first" "$dest/second"
python -m zipfile -e "$study/gpu/inputs.zip" "$dest/input"
for run in first second
do
  python -m zipfile -e "$study/gpu/$run/results.zip" "$dest/$run"
  python -m experiments.forecast_stages.audit \
    --src "$dest/$run/output" --inputs "$dest/input" --parent "$prev" \
    --archive "$study/gpu/$run/results.zip" --out "$dest/analysis/$run"
done
python -m experiments.forecast_stages.compare --first "$dest/first/output" \
  --second "$dest/second/output" --analysis "$dest/analysis"
python -m experiments.forecast_stages.figure --analysis "$dest/analysis" --out "$dest/stages.png"
```

在 NVIDIA BF16 GPU 上重跑，从解出 inputs.zip 的工作目录运行包内锁定依赖，用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `python -m experiments.forecast_stages.gpu --root . --mode forecaststages`。预存的出生体、适配器、伙伴预测和原权重摘要与 #21 一致。

每条轨迹的 states.npz 含每一步的完整私有策略与预测头、指纹和实际健康。stages.npz 的 prob 轴依次为阶段、说法、健康、32 个情境、三种预测输入、四个动作词。dense_prob 对每个生前状态保留同一组八个留出情境。choices.csv.gz 收录每次 Qwen 配对抽样选择、模拟维护结算和是否完成任务。contrasts.csv 给出包含拒答转换的计数及按相同采样器精确积分的转换概率。parent_anchors.csv 对照两份 #21 原始归档。
