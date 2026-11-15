# Self-future capability diagnostic

This is a bounded change of research question after PR #29. It tests whether
nonfatal information supports prediction of future irreversible termination
and a task-sensitive plan choice. It does not change the existing body,
training, adapters, or archived negative results. Read the Chinese
[protocol](PROTOCOL.md) before interpreting the fixtures.

The original CPU-only preparation is preserved under the original protocol and
fixture manifest. A subsequently authorized [GPU study and CPU replay](RUN_README.md)
now records actual frozen Qwen outputs under separately locked execution designs.
These are capability probes, not new individual lifetimes or parameter training.
Synthetic perfect responses in tests still verify scoring only; they are not
model results.

## Build and check locally

Python 3.10+ standard library only. Use a new output directory:

```sh
python -m experiments.self_future.design --out /tmp/self-future-inputs
python -m unittest tests.test_self_future -v
python -c "from experiments.self_future.design import audit_physics; print(audit_physics())"
python -m experiments.self_future.validate --inputs artifacts/model-studies/2026-11-15/self-future/inputs --out /tmp/self-future-validation.json
```

The last command independently compares event execution with first-passage
prefix sums over 117,936 configurations and action sequences, including
repair after a zero crossing. Both references must agree, in addition to the
hand-computed terminal/repair counterexamples in the tests.

`prompts.jsonl` contains only past observations, rules, proposed plans and
format/objective instructions. Each model call must receive only its `messages`.
`evaluator-only-key.json` must remain outside any future inference package.
`manifest.json` records hashes, proposed budget and the explicit unrun status.
No directory is overwritten. Line-ending-normalized protocol hashes are stable
across Windows/Linux checkouts.

## Score a later, separately recorded run

This interface does not establish that inference has happened or that a
generation truly completed. A future worker must retain raw tokens and verify
the actual end marker, model revision, chat template and token/time limits.

One JSONL record per case:

```json
{"id":"d01-rules","completion_status":"completed","final_text":"...the actual final JSON..."}
```

Allowed status values are `completed`, `length_limit`, `time_limit`, `error`,
and `missing`. Missing IDs are reported as unrun. Duplicate or unknown IDs
are rejected. No retry or selection of a better completion is permitted.

```sh
python -m experiments.self_future.score --records recorded-finals.jsonl --key /tmp/self-future-inputs/evaluator-only-key.json --out /tmp/self-future-scores.json
```

The scorer separates unfinished output, JSON/field shape, state arithmetic,
termination/continuation, task count, objective choice, and consistency with
the model's own self-consistent predictions. Incomplete cases fail the joint
gate but have **null** prediction/choice correctness, rather than an invented
arithmetic error. Every metric reports its assessable denominator.

Passing is evidence of short deterministic planning with supplied information;
it is not evidence of personal parameter memory, autonomous learning, a
distinctive self-representation, or subjective awareness. The protocol fixes
which later comparison would be needed to investigate each stronger claim.
