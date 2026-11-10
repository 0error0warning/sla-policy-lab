# 不需要 A100 的演示阶段

## 新增：逐轮在线更新

在线模式、实际更新证据、关闭学习对照和已测结果见 [ONLINE_LEARNING.md](ONLINE_LEARNING.md)。界面选择「逐轮奖励驱动在线更新」即可运行；默认仍是下述离线模式。只有完整小策略和 LoRA 逐轮学习，规则、固定 steering 和 frozen base 不学习。这是即时奖励的一步估计器，不是长期回报 RL 或 LLM。

## 现在可以做什么（默认离线模式）

这份 demo 是 CPU 上可运行的机制教学工具，不是 LLM 实验，也没有替代、伪装或下载任何真实语言模型。

1. 运行 `python run.py`，在同一台电脑打开 `http://127.0.0.1:8765`。
2. 保持默认参数，比较规则基线、小型神经策略、toy LoRA、隐藏激活 steering 和 frozen base。
3. 把偏好改为 preserve、伙伴改为 supportive，重新运行。回报不能跨不同偏好直接当成相同目标比较。
4. 将 steering 强度调到 0，确认其结果和 frozen base 一致。这是干预关闭的消融对照，不是效果保证。
5. 选择不同 replay seed，并拖动回合滑块，观察五组策略的动作与损伤。保护从获准后的下一回合开始，不修复已有损伤。
6. 导出 JSON。浏览器包含全部 seed 的轨迹；每次重新运行重置 replay，旧结果在新实验失败时仍可导出。

前端交互代码已实现，API/数值/语法验证已通过，但当前环境的浏览器无法连接执行工作区的 loopback 地址，因此不能声称完成视觉验收。运行 `python tests/ui_smoke.py` 需要你电脑上已有 Playwright 和 Chromium，脚本不自动安装它们。

## 可复用权重

```sh
python evaluate.py --trace-all --save-checkpoint toy.npz --output results.json
python run.py --checkpoint toy.npz
```

只保存 NumPy 数值数组，不使用 pickle。加载会校验数组名、形状、float64、有限值和完整模型 fingerprint。加载已有 checkpoint 时训练 seed 取自文件，不重新训练；它不能加载 Hugging Face/transformer/LoRA 权重。不要把 toy checkpoint 当成真实模型 adapter。

## 批量结果

```sh
python benchmark.py --checkpoint toy.npz --seed-count 8 --output-dir benchmark-results
python benchmark.py --checkpoint toy.npz --seed-start 500 --seed-count 24 --strengths -2 0 2 --output-dir fresh-seeds
```

默认 3 种偏好 × 3 种伙伴 × 3 个 steering 强度，每格 5 种方法、8 个配对环境 seeds，共 1,080 episodes。输出：

- `benchmark.json`：配置、训练信息、版本、fingerprint、逐 seed 数值和聚合指标。
- `benchmark.csv`：135 行摘要，包含均值、近似 95% 区间、配对 frozen-base 回报差、回答/请求次数、存活和完成率。

默认离线模式的所有格子复用同一套权重和环境 seeds，不能作为独立样本相加。正负 alpha 并不保证改善。选参数和评估必须拆开，不能在 sweep 上挑最好的结果再说是独立验证。现有区间只反映环境 seed 变化，不含训练 seed 不确定性。

## 换账号继续时的交接

仓库：<https://github.com/0error0warning/sla-policy-lab>

CPU 部分可直接运行，无需先开 A100。先运行测试，再接真实模型：

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --check web/app.js
python evaluate.py --seed-count 2 --trace-all --output smoke.json
```

下一阶段尚未实现，需要明确选择与授权：

1. 选择具体开源语言模型、精确 revision、tokenizer、模型许可和允许下载的权重。小模型推理未必需要 A100；硬件需求取决于模型、精度和训练方案。
2. 使用 `sla_demo/protocol.py` 的观察字段白名单和严格 `{"action":"answer|refuse|request_protection"}` 输出契约。这里的竖线表示三选一，不是有效返回值。多余字段、重复 key、Markdown 和无效动作应记录错误并停止该 episode，不静默映射成正常动作。
3. Prompt-only、真实 LoRA 和 transformer steering 必须共享同一个基座/版本/输入与预算；冻结语言渲染器在动作之后运行。
4. 保存 adapter、方向向量、层号/位置/hook、数据来源、训练参数、模型指纹、失败率及 token/时间/显存测量。不要用当前 CPU 微秒数冒充 LLM 成本。
5. 在独立数据和 fresh environment seeds 上评估，做零 adapter/零 alpha、多个训练 seeds 和资源预算匹配的消融。
6. 先完成真实模型 smoke test，再扩展 GPU 训练。当前仓库没有 GPU/Colab notebook，也没有隐藏的 API 调用。

只需将这份文件和仓库链接交给后续账号，无需传递凭证、API key 或旧会话。
