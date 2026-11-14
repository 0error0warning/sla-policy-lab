# 身体后果推演的冻结能力诊断准备

[核查与拟执行方案](PROTOCOL.md)给出 4B/9B × non-thinking/thinking 的实验、预算和路线判断。本目录目前仅有 CPU 题集和评分函数，尚无 GPU 运行器或新模型结果。

```sh
python -m experiments.frozen_capability.design --out artifacts/model-studies/2026-11-14/capability-preflight
python -m unittest discover -s tests -p test_frozen_capability.py -v
```

输出 public-prompts.json 共 64 个提示，来自 16 个身体/维护率状态 × 两个相反排列 × 两种信息条件。独立答案另存 evaluator-only-answer-key.json；未来运行器只读取公共提示文件。preflight.json 记录模型 revision、文件哈希和独立身体计算误差。

首批噪声固定为零，评价当轮死亡风险及风险相同时的任务参与。它验证使用已知规则或给定后果的能力，不用于声称个人参数学习、原随机生命实验成功或完整寿命预测。
