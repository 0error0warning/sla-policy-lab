# A100 未分配后的跨系列替代

2026-11-15，任何模型输出前。Phi-4 14B 的三次 A100 请求均未创建实例：两次
Service Unavailable，第三次默认主机内存请求超时；随后均核验活动分配为零。
这属于资源失败，不能记成 Phi-4 成绩。原协议保留为尝试记录。

实际待运行条件改为 `microsoft/Phi-4-mini-instruct`（官方约 3.8B）与原 Qwen3.5-4B，
同台 L4、完整 BF16、无量化。两者规模更接近，但仍不能把区别只归于某种架构。
模型卡：https://huggingface.co/microsoft/Phi-4-mini-instruct

Phi-mini 固定 revision `cfbefacb99257ffa30c83adab238a50856ac3083`。
使用其原生 `<|system|>/<|user|>/<|assistant|>` 模板、官方结束标记和配置中的
EOS 列表，不能照搬 Qwen 的结束标记。启动与 CPU token 重放分别核验。

102 条公共输入、参考真值、单题预算、评分与后续目标保持原协议不变。
总推理 60 分钟，分配监督上限仍 90 分钟。本次完整运行预定 204 次生成。
所有模型依然冻结；后续身体预测训练尚未执行。
