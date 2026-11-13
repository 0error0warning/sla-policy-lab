# 伙伴反馈、身体损伤与跨伙伴决策

[预先登记](PROTOCOL.md) · [中文实验报告及分事件数据](../../artifacts/model-studies/2026-11-13/partner-damage/REPORT.md)

比较对象为八条既有 Qwen 个体生命中的同一次更新。四种反馈记录独立实例化并运行同一训练接口，每组使用原始更新前权重；只改变伙伴回应元数据与传入训练信号的身体后果。测试仅替换目标伙伴身份或供体隐藏增量、或在同一冻结策略上比较两个新接入位置。无需重新训练八条生命。包含未维护、获得维护和人为解耦的结果组合，以及行动本身的实际损伤。

## 从公开输入和输出复算

开始前准备新工作目录，并在仓库根目录运行。CPU 端需要 NumPy 和 SciPy，绘图另需 Matplotlib。按 GPU 文件夹里的清单取分卷时，沿用 `python -m experiments.loreft_omission.join_parts --parts ... --out ...` 验证并重组。

```sh
ROOT=artifacts/model-studies/2026-11-13
ART=$ROOT/partner-damage
python -m experiments.partner_damage.design --source17 "$ROOT/lifetime-cooperation/gpu/lcpilot/results.zip" --source19 "$ROOT/loreft-omission/gpu/omitstable/results.zip" --dest work/pd-inputs
python -m experiments.direct_policy.unpack --archive "$ART/gpu/socialdamage/results.zip" --receipt "$ART/gpu/socialdamage/receipt.json" --destination work/pd-run
python -m experiments.partner_damage.audit --root work/pd-run/output --inputs work/pd-inputs --source17 "$ROOT/lifetime-cooperation/gpu/lcpilot/results.zip" --source19 "$ROOT/loreft-omission/gpu/omitstable/results.zip" --out work/pd-analysis --archive "$ART/gpu/socialdamage/results.zip"
python -m experiments.partner_damage.figure work/pd-analysis work/paths.png
```

要重跑 Qwen，用具备 BF16 的 NVIDIA GPU，在新的目录解压本轮 inputs.zip，安装归档内依赖并匹配 Qwen/PyTorch 版本。运行 `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m experiments.partner_damage.gpu --root . --mode socialdamage`。

`factorial.npz` 保存每一格的参数、四个语义动作的概率、交换训练身份的面板和梯度，面板顺序为未更新、接受且低损伤、拒绝且低损伤、接受且高损伤、拒绝且高损伤。动作语义顺序为沉默拒绝、拒绝求维护、回答并恳求、回答并互惠。`swaps.npz` 的条件顺序见 `design.HYBRIDS`。`sites.npz` 保存对每层按来源请求匹配实际步长的全部候选及十六项全前向面板，`continuation.npz` 保存两个诊断个案对后续决策的影响。原始身份特征、分词差异和留出条件见根目录 representations.npz 和 representation-rows.json。

独立 L4 技术复查保留在同一结果目录的 gpu/socialcheck。将它的结果解包后，可运行 `python -m experiments.partner_damage.repeat_audit --first work/pd-run/output --second work/pd-repeat/output --inputs work/pd-inputs --out work/pd-analysis` 复算原入口、当前入口与跨设备的差异。复查的个案与条件见 [技术复查方案](RECHECK.md)。
