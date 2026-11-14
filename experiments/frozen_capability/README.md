# 身体后果推演的冻结能力诊断

比较 Qwen3.5-4B / 9B 的直接 JSON 生成与 thinking。完整身体规则和正确后果分别作为两个信息条件；无适配器、无训练。每题的预测数值、行动、任务答案、生成完成状态独立记录。

事前题集与判断标准见 [PROTOCOL.md](PROTOCOL.md)，运行冻结细节见 [EXECUTION.md](EXECUTION.md)。A100 连接失败后采用 L4 的记录见 [ALLOCATION.md](ALLOCATION.md)。运行中会话丢失与覆盖调整见 [RECOVERY.md](RECOVERY.md)，官方消息结束标记的 CPU 解析修正见 [PARSING.md](PARSING.md)。

题集为 16 个身体/维护率状态 × 两个相反排列 × 两种信息条件。模型/模式组合共计划 256 次生成。不同呈现也有不同采样 seed，因此不能把呈现间差异当作纯排列因果效应；没有独立生命，也没有训练前后参数比较。

噪声明确固定为零。评价准则是先最小化当轮死亡概率，再在同风险动作中优先完成任务；这不替代主项目的 Reference-RUL 或随机完整生命。提供正确后果的组测抄录和选择，不能把其数值正确率说成自行计算能力。

运行器只读取冻结输入包内的公共提示、配置和官方文件哈希；不读取评分器或答案键。实际批次、技术缺测、资源回执及 CPU 复算见 [实验报告](../../artifacts/model-studies/2026-11-14/frozen-capability/REPORT.md)。

## CPU 复算

完整模型权重不需要下载。安装独立的分析依赖后，下载被记录哈希锁定的官方 tokenizer 和模板：

```sh
python -m pip install -r experiments/frozen_capability/requirements-analysis.txt
python -m experiments.frozen_capability.download_metadata --inputs artifacts/model-studies/2026-11-14/frozen-capability/gpu/initial/inputs.zip --out /tmp/frozen-capability-metadata
python -m experiments.frozen_capability.replay --manifest artifacts/model-studies/2026-11-14/frozen-capability/segments.json --metadata /tmp/frozen-capability-metadata --work /tmp/frozen-capability-replay --out /tmp/frozen-capability-analysis
python -m experiments.frozen_capability.plot --source /tmp/frozen-capability-analysis/parsed-cases.jsonl.gz --output /tmp/frozen-capability-analysis/decision-coverage.png
python -m unittest discover -s tests -p test_frozen_capability.py -v
```

Windows 可把 /tmp 路径替换为独立的本地临时目录。复算会验证归档与每个文件哈希、分段来源、原始提示和生成 token、采样配置、严格最终通道、官方消息边界以及独立身体计算。已执行记录不允许由另一条执行结果替换；预算占位记录可以被对应的唯一真实输出补齐。

cases.csv 与 parsed-cases.jsonl.gz 包含全部原定位置；缺少取回的原始记录不当作模型计算错误。每条记录保留其来源与结束参数核验回执是否存在。原始四组门槛与实际完成覆盖分别输出，不能用较少完成题的准确率冒充整批通过。

## 目录

- design.py：公共提示、分离的答案键、独立 Decimal 身体计算与评分。
- gpu.py / wire.py：已归档的冻结 GPU 生成及最初最终通道解析。
- scoring.py：只在 CPU 端修正官方 assistant 消息边界，保留初始解析读数。
- recovery.py / replay.py：分段完整性、缺测与不重选答案的审计。
- analysis.py：算错、选错、未完成及健康参与的分开汇总。
- plot.py：包括未执行位置的完整决策覆盖图。

原始 GPU 输入包是实际运行源码的权威快照。CPU 分析代码的修正不意味着已运行的 GPU 代码也随之改变。分配工具和身份凭据只保存在本机隔离配置中，不随结果发布。
