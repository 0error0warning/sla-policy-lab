# 固定配置与官方建议的边界

在本轮运行中核对了同一固定权重 revision 的官方模型卡：[4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md#best-practices)、[9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#best-practices)。通过 Firecrawl read-docs 获取，2026-11-14；不是只凭模型名称推测的默认值。

本轮承接 #24 的固定采样配方：temperature=.6、top_p=.95、top_k=20、presence_penalty=0、repetition_penalty=1。两模型两模式一致。这套配方在模型卡对应精确代码类 thinking 任务；一般 thinking 建议 temperature=1、top_p=.95、top_k=20、presence_penalty=1.5。模型卡还给出其他 non-thinking 配方，不能把某一个配方称为所有模式唯一推荐值。

官方建议多数查询允许 32768 token，复杂数学/编程基准可使用更高上限。本轮的 8192 是明确的成本上限，吞吐预检证明在墙钟内能实际使用它，但不证明它对每个问题都足够。官方建议也不证明一定需要用满那么多 token。

不在已经取回答案后改 sampler 或增加某一模型的输出上限；保持本轮可比，所有未完成均单列。若本轮在 8192 token 内仍缺最终答案，不能从这种有限输出配置推导出模型无法理解身体规则，也不能宣称已测到最优配置下的能力上限。后续若比较官方一般 thinking 配方，应另登记为新条件，并保留同题、同预算及旧输出，而不能把替换后的结果覆盖本轮。
