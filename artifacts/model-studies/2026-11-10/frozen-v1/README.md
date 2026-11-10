# Original frozen-model screen

[Report](REPORT.md) · [Scores](results/scored-results.json) · [Per-question scores](results/case-scores.json)

- `inputs/`: the fixed public questions, separate answer key, model candidates/revisions and pre-execution manifest.
- `raw/model_0`: Qwen3.5-4B; `model_1`: Qwen3-4B-Instruct-2507; `model_2`: Qwen3-1.7B.
- Each model retains its original qualification/rollout JSONL, independent terminal-control-normalized JSONL, loading information, runtime, parameter-freezing checks, warmup, and checkpoint weight hashes.
- `executed/gpu_job.py`: exact source used on the GPU. Use the repository reproduction entrypoint, not this archived file directly.

48 calls/model represent 24 problem units with English/Chinese repetitions. Eight lives/model use two shared event-tape seeds across four environment cells. Protocol failures are distinct from environment deaths. All 24 lives were independently replayed. No training occurred.

The score remains 18/48, 24/48, 27/48 for 1.7B, 4B-Instruct, 3.5-4B respectively. The separate diagnostic study does not replace these scores.
