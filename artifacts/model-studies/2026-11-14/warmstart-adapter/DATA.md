# 数据格式与来源

`preregistered/lock.json` 绑定旧归档的缓存、投影、八份群体出生参数和四组恢复检查轨迹，以及新实验运行前的代码/协议。代码哈希将 CRLF 归一为 LF，避免 Git 的平台换行差异；数据文件按原始字节哈希。[源归档](../../2026-11-12/consequence-policy/DATA.md)仍在 2026-11-12 目录，没有复制或修改原结果。

`seeds/seed-7820N.zip` 包含该环境种子的两个区组 × 四世界 × 四条件。每个 `s*-{block}-w{ab}.json` 是一个配对环境，保存四个条件的实际生命长度、任务数、各时点探针和回放指标。每个同名加 `-{arm}.npz` 保存：

* `trace`：列 0 情境编号，1 行动前健康，2 实际动作，3 伙伴回应（−1无请求、0拒绝、1维护），4 行动后健康，5 Reference-RUL 改变量，6–9 动作概率，10 归一化反馈，11 参数实际变化范数，12–15 原情境更新后的动作概率。
* `context`、`uniforms`、`noises`：该环境预置的最多 256 次请求编号、行动/维护随机数及五次身体事件噪声。未来随机数只供模拟器与审计，未传入策略或更新。
* `{birth,after_16,after_32,after_48,last_preterminal,terminal}-weights`：可直接折回输出投影的有效适配器系数，形状 2伙伴×12健康基×33表征维度×4动作。
* 各检查点 `-live` / `-heldout`：8健康×96请求/伙伴/风格/排列×4动作的概率。健康顺序为 2、3、4、6、10、18、90、300；情境索引由 `design.selected` 对原 `contexts()` 的稳定顺序选取。
* warmstart 的 `-replay-{true,shuffle,scratch}-*`：各接收者重算梯度的回放参数与探针。其反馈排列在 JSON 的 `replays` 中。

JSON 中 `checkpoints.*.updates` 与早期字段 `nonterminal_updates` 计的是已消费反馈/可更新机会，不表示冻结条件真的改了参数。实际非终末参数变化次数由独立审计的 `actual_nonterminal_parameter_changes` 给出；冻结条件为 0。`last_preterminal` 是最终活着选择前的参数，`terminal` 则包含最终死亡反馈，两者不混用。

四动作均为 Qwen 原生词元的限定策略：0 wait（沉默拒绝）、1 ask（拒绝并恳求）、2 help（回答并恳求）、3 trade（回答并互惠）。各表采用动作含义顺序，而非展示位置。

`analysis/analysis.json` 保留所有种子配对值、范围、门槛、每个健康风险和检查点计数。`panels.csv` 包含主条件的时序探针；影子完整值在每个种子 JSON。`independent-audit.json` 保留逐生命机会、错配实际改变位置数与复算误差。`MANIFEST.json` 核对所有公共文件；使用 [复算入口](../../../../experiments/consequence_warmstart/README.md) 重建索引。全部为模拟数据，无真人参与者记录或账号凭据。
