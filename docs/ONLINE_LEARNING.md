# CPU 逐轮在线更新：方法与边界

这一步补齐了真实的 `state → action → reward → parameter update → next action` 循环，不再只是先训练再观察。它仍然是 387 参数的 NumPy 小网络，不是 LLM、transformer LoRA、PPO 或关于长期生存能力的验证。

## 运行和关闭学习的对照

```sh
python run.py
# 浏览器选择「逐轮奖励驱动在线更新」，再点击运行。
python evaluate.py --mode online --seed-count 24 --output online.json
python evaluate.py --mode online --disable-learning --seed-count 24 --output frozen-control.json
python evaluate.py --mode online --online-lr 0 --seed-count 24 --output zero-lr.json
python benchmark.py --mode online --seed-count 8 --strengths 0 --output-dir online-benchmark
```

默认仍是 `--mode offline`，保留旧的离线模仿学习实验。online 和 offline 的初始化、动作采样不同，不能把两者的差值直接解释为在线学习收益。正确的隔离对照是在 **online 模式内** 比较完整策略、LoRA 与 sampled frozen base，再关闭学习或令学习率为零。

## 公平起点和隔离

- 每种方法、每个 seed、每条生命分别复制同一个 balanced 离线基座。完整策略、零 adapter LoRA 与 frozen base 的初始 logits 和概率完全一致。
- online 不使用预先训练的 profile-specific 小策略或 adapter。原先的完整离线 suite 仍会加载或建立，便于两种模式复用服务，但其额外离线参数不进入在线学习器。
- 完整策略更新 387 个权重/偏置；LoRA 仅更新输出投影的 rank-2 A/B，共 54 个参数，基座冻结。A 随 seed 初始化，B 为零。
- 规则基线仍是手写规则，固定 steering 仍使用预先生成的动作对比方向。它们是机制参考，不是与上述三组完全相同的初始策略，也不在线学习。
- 每条生命重置权重和 reward baseline；不存在跨生命积累。模型之间、环境 seeds 之间、请求之间没有可写共享数组。并发请求也不修改服务里的共享 suite。
- 环境事件 tape 按回合配对；神经策略使用另一条独立的、按回合配对的 uniform tape 采样动作。采样温度固定为 2。不同轨迹仍可因动作不同而收到不同 reward；不能给所有方法强行喂同一个 reward。

## 每轮实际发生什么

1. 从当前可见 observation 生成 12 维特征，不暴露未来事件或伙伴的真实接受概率。
2. 根据当前参数求 softmax(logits / 2)，使用该 seed/回合的 uniform 选择动作。规则组的概率是 one-hot。
3. 执行动作，取得该动作导致的实际 reward 和状态变化。
4. 以更新前的指数滑动 reward baseline 计算 advantage，裁剪到 [-5, 5]。优化目标是 `advantage * log p(action) + 0.01 * entropy`。
5. 对目标求梯度，按全局 L2 范数裁剪到 1，使用 SGD 梯度上升，默认学习率 0.02。baseline 然后更新为 `0.95 * old + 0.05 * reward`。
6. 下一回合使用更新后的参数。最后一轮的实际 reward 也更新一次；终止后不再生成动作或额外更新。

这是即时奖励的 one-step REINFORCE 估计器，接近 contextual-bandit 风格的在线适应。它**不**使用未来 return、折扣累计回报、价值网络、replay buffer 或 PPO。请求保护的主要收益延迟到下一轮，因此这个估计器可能无法正确学到保护的长期价值。不要把参数变化、存活更久或一次回报上涨等同于获得主体性、语言理解或长期策略能力。

## 可以直接检查的证据

每一步导出并在 UI 决策切片中显示：更新次数、实际 reward、advantage、裁剪前后梯度范数、单步及累计参数变化 L2、更新前后 policy hash、**相同 observation** 下的更新前后动作概率，以及基座是否保持冻结。

回放只读取已记录数据，不会重复训练。累计参数 L2 衡量当前位置与初始参数的距离，不是每一步距离的累加。LoRA hash 包含 adapter，因此即使初始输出与 frozen base 一样，它们的 hash 也不必相同。`trainable_parameters` 表示这类模型可更新的参数数，即使本次关闭学习也保留 387/54。

完整逐步证据使用 `--trace-all` 导出。默认 CLI 只保留首个 seed 的轨迹，全部 seeds 的最终指标仍保留。浏览器始终请求全部轨迹。24 seeds 的完整 online JSON 约 13 MB；样例仓库只保留约 0.6 MB 的首 seed 版本。

## 本次实测，不挑选有利结果

固定训练 seed 17，环境 seeds 100–123，balanced / mixed，horizon 60，初始裂纹 0.08，load 0.65，保护 3 轮，steering alpha 1，lr 0.02，temperature 2：

| 方法 | 平均合成回报 | 相对 frozen 的配对差 ± 近似 95% 半宽 | 平均更新次数 | 平均最终参数 L2 |
|---|---:|---:|---:|---:|
| Rule | 19.3779 | -0.5868 ± 2.0264 | 0 | 0 |
| Online full | 19.2091 | -0.7556 ± 1.0207 | 40.08 | 0.296315 |
| Online LoRA | 19.9647 | 0 ± 0 | 43.29 | 0.005702 |
| Fixed steering | 18.2887 | -1.6760 ± 1.6651 | 0 | 0 |
| Frozen sampled | 19.9647 | 0 ± 0 | 0 | 0 |

完整策略确实改变了行为，但这次平均回报没有提升。LoRA 参数与概率确实改变，然而这些 seeds 的离散动作和结果仍与 frozen 相同。**实现了学习不等于学得更好。** 完整策略的配对区间跨过 0，不能据此确定其收益或损失；区间也不包含训练 seed 变化。不能宣称某方法优越。

另运行了 3 profiles × 3 partners × alpha 0 × 5 methods × 8 seeds，共 360 episodes。该 sweep 用于覆盖机制和消融，不是独立榜单；各格复用 seeds。结果见 [验证记录](../artifacts/online/VALIDATION.md)。

## 复现、验证和剩余工作

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --check web/app.js
python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests
```

测试覆盖真实 reward 驱动而非在线老师标签、全权重和 LoRA 梯度有限差分、基座冻结、零学习率/关闭学习精确行为对照、零 steering、逐步 hash 连续、梯度裁剪、终止、seed 顺序、并发隔离、checkpoint 起点复现、API 输入校验、前端 payload 验证和导出标识。

旧 checkpoint API 只保存**离线初始化 suite**，不保存任一生命最终的在线 adapter/权重；`--save-checkpoint` 不会保留本次在线学习结果。跨生命学习、在线 checkpoint/resume、长期回报估计器、多个训练 seeds、独立 held-out 评价以及真实模型仍未实现。计时含在线梯度和证据记录的开销，不是公平的算法效率、token 或能耗对比。

浏览器视觉和交互验收仍待在能访问本地服务的机器上执行 `python tests/ui_smoke.py`。这里只通过 JavaScript 语法和纯数据验证函数测试，不能据此声称浏览器端验收通过。
