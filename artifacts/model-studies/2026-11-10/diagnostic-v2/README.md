# Staged diagnostics and new seeds

[Report](REPORT.md) · [Execution-before specification](SPEC.md) · [Full scored results](results/scored-diagnostics.json) · [Per-question scores](results/case-scores.json)

- `inputs/`: 48 diagnostic questions, separate answer key, fixed model revisions, conditions, and 32-life matrix. The answer key is published for verification but is excluded from prepared GPU inputs.
- `raw/model_0`: Qwen3.5-4B; `model_1`: Qwen3-4B-Instruct-2507. These files belong to the **complete recovery**.
- `results/rule-baselines.json`: all 128 CPU rule-policy lives (four policies × 32 lives).
- `interruption/`: preserved first-attempt generations. Their equality to the complete recovery is recorded in `results/first-attempt-comparison.json`. They add zero independent samples.
- `executed/gpu_job.py`: exact measured worker source; use the separate reproduction entrypoint.

D2-G is non-thinking/greedy with 128 output tokens. D2-T is the separate Qwen3.5-only English subset with thinking/greedy and 1024 output tokens; all eight cases exhausted the budget without a final answer. D2-R adds eight new seeds, each repeated across four paired environment cells. D2-J (constrained JSON generation) was reserved and **not executed**.

Four physical counterfactual scenarios, A/B label swaps, and language repetitions are not 48 independent samples. Rollout uncertainty is grouped by eight seeds, not 32 independent lives. Already-computed future returns are supplied only in the explicit counterfactual comparison stage, never in the hidden-future rollouts.

The GPU writer serialized empty loading-key sets as the literal string `set()`. The verifier recognizes exactly this empty marker; nonempty values still fail. The raw files are unchanged. This serialization issue did not require model reruns or score changes.
