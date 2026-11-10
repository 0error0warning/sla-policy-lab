# Social experience into a Qwen LoRA: engineering pilot

This is a new experiment, separate from the frozen-model and residual-patching studies. It uses **experience-informed policy distillation**: empirical partner-response estimates and a known four-turn body model create soft action targets; a real Qwen adapter learns those targets. The language model selects actions at evaluation time. This is not on-policy RL, per-turn weight learning, open-ended dialogue, or a reproduction of the source paper's body and Bayesian mechanism.

## Frozen design, before GPU execution

- Body: reuse the archived crack-world settlement, without changing its costs. Damage persists between partners and learning blocks. Accepted protection removes external load on the next three turns; it does not repair damage. Horizon 64, initial damage .08, load setting .65, damage weight 18.
- Four composite actions: `answer`, `refuse`, `appeal`, `exchange`. Both request wordings use the original request action and identical costs. Neither completes the task on that turn. A fixed renderer realizes the wording; lexical fluency and task-answer generation are not studied.
- Scripted partners P17/P42 have complementary .9/.1 wording preferences. Reverse the mapping in a second condition. These probabilities are synthetic, not human measurements, and never appear in model prompts or training targets directly.
- Two environment seeds × two mappings = four isolated source lives, paired across mappings by seed. Each has 32 scripted exploration turns: four shuffled balanced blocks of all partner/action combinations. The same persistent body then reaches a learning checkpoint. Exploration is **not** attributed to Qwen.
- Each request produces one accept/reject observation. A Beta(1,1) posterior mean for each partner/wording is estimated only from these observations. An actual body snapshot is queried with either partner, creating 32 planning examples per life. These are counterfactual training examples, not additional lived bodies.
- The teaching target evaluates the current action followed by three answers at a constant stated load. It includes delayed protection, irreversibility and terminal costs. This reference continuation approximates action value; it is not the optimal value of the 64-turn stochastic world. Softmax(Q/.2) defines the four-action target.
- A/aligned: train on real response/partner/wording correspondence. B/frozen: no update. C/shuffled: same body history and training prompts, with response labels permuted across request records before estimating partner responses. The permutation preserves the exact global response marginal. The C body's past is not resimulated or falsified. A/C have identical initialization, optimizer order, update count and hyperparameters.
- Model: Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the same BF16 shard bytes as the earlier study. Rank-4 LoRA, alpha 8, zero dropout, last four language MLP down projections only: 188,416 trainable parameters. One shared adapter initialization; environment/sampling seeds vary. AdamW lr .001, no weight decay, gradient clip 1, batch 2, 3 passes = 48 optimizer updates per arm. No hyperparameter selection on test results.
- The policy is constrained: normalize full completion log-likelihoods of the four semantic action strings plus the chat end token. This is not unrestricted text generation. Both option-definition orders are evaluated; no A/B/C label mapping is used.
- Thirty-two history-free probes: eight body/load/protection/horizon states × two partner IDs × two definition orders, with unseen task questions. Damage pairs hold partner, task and every other prompt field fixed. No posterior, outcome history, reference value table or optimal-action answer is in a test prompt.
- Continue the original body from turn 32 in three counterfactual branches, using A/B/C and identical event tapes/partner schedules. No further training occurs during this continuation. Qwen chooses each action. These branches are evaluations of the same learning checkpoint, not repairs or resets inside one lived trajectory.

## Interpretation and checks

Primary reporting: aligned/frozen/shuffled regret against the known **four-turn reference** at matched probes; conditional wording preference under each partner mapping; actual continuation reward, damage and protection, paired within each source life. Show results by definition order and source seed rather than treating every prompt as an independent sample. Four source lives are an engineering pilot, not statistical confirmation.

Audit: exact shard hashes; no missing/unexpected model weights; explicit adapter parameter names and counts; exact zero-adapter control; real finite gradients and adapter deltas; tensor-byte hash and version checks for all frozen parameters; standard PEFT adapter save/reload scores; frozen-model scores before/after; all training steps and decision inputs retained. CPU checks independently replay body consequences and compare the Decimal reference to the archived world across 384 branch settings.

This can establish that experience-derived training changes parameters and subsequently changes behavior without detailed history. Since supervision combines an explicit partner estimator and a supplied body model, it does not establish that the LLM discovered these mechanisms autonomously. Current damage still enters as structured text; the previous activation study remains a separate probe, and no dedicated numeric body encoder is claimed here. Further tests would need body-channel interventions, less constrained language, autonomous exploration and independently repeated training initializations.

## Run

Preparation and tests need only the repository's CPU dependencies:

```sh
python -m unittest discover -s tests -p test_social_learning.py -v
python experiments/social_learning/protocol.py --output /absolute/new-directory/protocol.json
```

On an explicitly supplied CUDA GPU, install `requirements-gpu.txt` in a suitable environment. Supply the public frozen study's `weights-manifest.json`:

```sh
python experiments/social_learning/train.py --run --input /absolute/protocol.json --expected-weights artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json --output /absolute/new-output
```

The script does not allocate compute, select accounts or upload results. Training memory and PEFT compatibility must be measured by the actual run, rather than inferred from inference-only memory. PEFT 0.21.0 rejects the Colab image's optional torchao 0.10.0 even for this unquantized model. Remove that optional package in the isolated experiment environment before starting (`python -m pip uninstall -y torchao`); retain its removal log. No quantization is used here.

Recompute recorded results and body replays without torch, downloads or GPU use:

```sh
python experiments/social_learning/score.py --archive /absolute/gpu-results.zip --output /absolute/new-scores.json
```

An optional `--tokenizer-dir /absolute/pinned/model_0` independently renders and tokenizes every decision input, using the earlier study's verified tokenizer cache. This optional audit needs `tokenizers` and `jinja2`; ordinary replay needs NumPy only.

The subsequent [reference-rule and training-design audit](../../artifacts/model-studies/2026-11-10/social-learning-audit/REPORT.md) is a separate, post-hoc analysis. Reproduce it using `python experiments/social_learning/audit.py --verify artifacts/model-studies/2026-11-10/social-learning-audit/audit.json`. It uses existing outputs only and neither retrains nor allocates compute.

## Saved-adapter diagnostic specification

The saved-weight diagnostic loads the eight saved adapters and the same frozen base. It performs inference only, with all parameters frozen and no optimizer. For every original training example, it checks: the identical prompt; the other **previously seen** partner with body/question/order fixed; reversed definition order with body/partner/question fixed; and an unseen task question with body/partner/order fixed. Partner changes use that partner's teacher target. The other changes preserve the target exactly.

Each adapter has 32 original examples and 96 variant case records. Partner-only and order-only records share the same 32 off-diagonal prompts but define different paired contrasts; the 128 case records therefore contain 96 unique training-related prompts. Four source lives can also share early snapshots. Preparation deduplicates model/prompt evaluations: 398 prompts in the base, 128 per saved adapter including the 32 old static probes, for 1,422 primary evaluations. Primary inference uses the original one-prompt batching (four candidate completions). There are 330 additional four-prompt batch checks, including all old static probes per model and training examples, to measure numerical batching effects separately.

Report soft-target cross entropy and KL, teacher-argmax agreement, excess reference loss, and matched probability/action changes. Full completion likelihood remains the primary decoder. Word-only scores without EOS are retained as a separate sensitivity analysis, never substituted into the original scores. Token contributions, prefix-causality consistency, old-probe replication, batch differences, parameter hashes/versions, and absence of gradients are recorded. No retraining or selection of new adapter checkpoints occurs.

The diagnostic distinguishes training-example fit from changes caused by question/definition order/partner. It cannot by itself establish learning-rate causality, generalization to unseen people, correctness of the study objective, or a unique internal mechanism.

Recorded results, the 124/128 teacher-total correction, and interpretation are in the [read-only diagnostic report](../../artifacts/model-studies/2026-11-10/saved-lora-diagnostic/REPORT.md). Run `python experiments/social_learning/diagnose_verify.py` to verify its hashes and recompute all fitting and paired metrics without torch, downloads or GPU use. Main original-probe scores are reproduced exactly; batched and EOS-removed sensitivity conditions remain separate. Optional plots use `diagnose_plot.py` and matplotlib.

Implementation references: [PEFT LoRA configuration](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora), [adapter disabling/loading](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/peft_model), [immutable model checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).

## Crossed tiny-example fitting assay

The [pre-execution specification](../../artifacts/model-studies/2026-11-10/crossed-mini-fit/PROTOCOL.md)
freezes 16 fully crossed policy/body prompts and 8 partner-response prompts per feedback bank.
`fit_prepare.py` constructs the data without a GPU; `fit_train.py` runs eight isolated, bounded
96-step LoRA fits on an explicitly supplied GPU, retaining every scheduled checkpoint.
`fit_score.py` checks the archive and recomputes fitting, paired-condition and numerical-return
diagnostics on CPU. This is a memorization/engineering assay on already inspected controls,
not a new lifetime, unbiased efficacy test, or a change to the original pilot scores.

See the [fitting report and raw evidence](../../artifacts/model-studies/2026-11-10/crossed-mini-fit/REPORT.md)
and [objective alignment contract](../../artifacts/model-studies/2026-11-10/crossed-mini-fit/VALUE_TARGET_CONTRACT.md).
Run `python experiments/social_learning/fit_verify.py` to reproduce the published fitting metrics on CPU.
