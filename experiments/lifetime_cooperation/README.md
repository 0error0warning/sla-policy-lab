# 有限生命中的合作与避险

[预先登记的实验设计](PROTOCOL.md) · [本次报告与数据](../../artifacts/model-studies/2026-11-13/lifetime-cooperation/REPORT.md)

每条 Qwen 生命使用相同的群体出生 LoRA，自己选择四个语义动作。环境按所选行动、伙伴维护和当轮噪声结算身体，接下来才更新私有参数。执行中每一个实际行动及其梯度都经过完整 Qwen。只读探针复用固定请求在冻结层的张量，逐检查点与全模型前向核对。探针不会扣除健康或改变该个体的学习记录。

## 从归档复算先导

在仓库根目录运行下列命令。CPU 分析需要 NumPy 和 SciPy。解包目标使用一个新目录。请参阅归档中的 `receipt.json` 了解本次 GPU 运行依赖和设备。

```sh
ART=artifacts/model-studies/2026-11-13/lifetime-cooperation
python -m experiments.direct_policy.unpack --archive "$ART/gpu/lcpilot/results.zip" --receipt "$ART/gpu/lcpilot/receipt.json" --destination work/lcpilot
python -m experiments.lifetime_cooperation.audit --root work/lcpilot/output --out work/lcpilot-analysis --archive "$ART/gpu/lcpilot/results.zip"
python -m experiments.lifetime_cooperation.exposures --root work/lcpilot/output --out work/lcpilot-analysis
```

从 GPU 输入归档复跑时，解出 `gpu/lcpilot/inputs.zip` 到新目录并在该目录执行。需要支持 BF16 的 NVIDIA GPU、与归档相同的 PyTorch 环境及 `requirements-gpu.txt` 中列出的依赖。Qwen 模型与出生参数均有 SHA-256 检查。worker 会下载或复用所需模型文件，校验后才载入权重。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.lifetime_cooperation.gpu --mode lcpilot --root .
```

每条物理生命的 `life.json` 给出任务、暴露和各个评估时点，`events.jsonl` 给出全部物理行动。上下文编号可在 `output/contexts.json` 查到实际文字和动作布局。学习组的 `parameters.npz` 含出生和末次生前检查点参数，Qwen 向量展开顺序为 A（2×12×33×4）再 B（4×2560），FP32。逐轮更新后哈希和步长见事件日志。末次选择后才致死的反馈不会进入生前参数。

`panels.npz` 的维度依次是参数条件（本生命、出生、错配）、措辞（原、留出）、健康（2、6、90、300）、32 个上下文和四个语义动作（沉默拒绝、拒绝并请求维护、回答并恳求、回答并互惠）。`_on_tape` 表示用 Qwen 完全相同的经历训练的简单策略。改变伙伴偏好的世界在第 25 次请求起仅改变 P86。主结论只采用稳定世界。
