# 连续偏好的更新遗漏实验

[事前登记](PROTOCOL.md) · [实验结果及逐种子归因](../../artifacts/model-studies/2026-11-13/loreft-omission/REPORT.md)

“第 t 轮”始终指第 t 次真实请求到达、模型尚未执行此次行动的决策时点。此时最多已获知 t−1 次反馈。单次、三次遗漏和两个单独遗漏分支从同一份正常参数开始。每条分支每轮看到相同的真实请求、伙伴、健康与后果；断开指定的参数更新后，继续按自己的参数重新计算以后每个训练梯度。所有生命在物理上按归档的同一历史前进。

每个生前参数快照产生健康 2、6、90、300 的面板。每个健康值有 #17 登记的 32 个留出措辞、平衡布局的请求（两个任务、两语气、两伙伴、四布局）。冻结的上游残差与末层传感路径和 #18 相同，实际训练每次调用完整 Qwen。只读面板复用冻结的上游计算；用完整前向逐项抽验。

在仓库根目录，用新目录解开结果并将 #17 与 #18 的公开结果作为 `source17.zip` 和 `source18.zip` 放入同一个目录。CPU 重算需要 Python、NumPy 和 SciPy。若 GPU 结果分卷，在解包前用 `python -m experiments.loreft_omission.join_parts --parts path/to/results.zip.parts.json --out path/to/results.zip` 校验并合并。工作文件使用新目录。

```sh
ART=artifacts/model-studies/2026-11-13/loreft-omission
python -m experiments.direct_policy.unpack --archive "$ART/gpu/omitstable/results.zip" --receipt "$ART/gpu/omitstable/receipt.json" --destination work/omitstable
python -m experiments.loreft_omission.audit --root work/omitstable/output --inputs work/omission-inputs --out work/omission-analysis --archive "$ART/gpu/omitstable/results.zip"
python -m experiments.loreft_omission.figure work/omission-analysis work/onset.png
```

GPU 独立重跑可将归档的 `inputs.zip` 解到新目录，按其中依赖文件和 GPU 收据匹配版本，用支持 BF16 的 NVIDIA GPU 运行 `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m experiments.loreft_omission.gpu --root . --mode omitstable`。输入归档已包含两份来源数据。

原始输出 `lives/s*/normal.npz` 保存正常分支每一步的 LoReFT 参数（R、W、bias 展平）、真实请求概率、出生策略概率及全量面板。`plan.json` 给出真实三连、固定面板三连、末期众数、阴性参照及普通窗口。`forks/u*/trace.npz` 在首个遗漏的决策时点之前共享正常前缀，内含之后逐步面板、真实请求及最终参数；同目录日志记录每步参数哈希和实际步长。四个语义动作依次为沉默拒绝、拒绝并求维护、回答并恳求、回答并互惠。
