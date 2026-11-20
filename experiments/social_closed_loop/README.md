# Social prediction in an A → B → A closed loop

The completed 2026-11-11 study, individual-life evidence, fitting diagnostics and post-hoc pooled control are in the [experiment report](../../artifacts/model-studies/2026-11-11/closed-loop/REPORT.md).

See [the frozen protocol](PROTOCOL.md) for all design choices and research limits, and [the P86 wording clarification](ERRATA.md). Each partner first presents a raw request. A common external controller uses a learner's response predictions and the current body's expected next Reference-RUL to select one of the paper's four composite actions. The resulting partner response changes both the body and the evidence used for subsequent learning.

This is a controlled, mechanism-inspired implementation. The Beta reference receives known incoming-framing categories; the neural conditions receive raw language. The common epsilon-greedy planner is not the source paper's Thompson sampler. The renderer is fixed, and Qwen predicts responses rather than generating answers or directly selecting actions.

The real model is Qwen3.5-4B at a pinned revision. Its 0–3-layer MLP LoRA and a new partner-routed six-feature readout receive the same hierarchical posterior-mean soft labels every five encounters. Feedback records are retained to calculate those labels. This is periodic online posterior distillation, not raw-reward RL or a demonstration that parameters replace all training-time evidence storage.

The routed readout has six coefficients per registered partner. Its group transfer comes through explicitly generated hierarchical targets, including targets for the unencountered partner; it does not independently discover the group rule or an arbitrary new identity. Optimizers and update budgets differ between methods despite matching target definitions.

Use an explicitly supplied CUDA GPU with BF16 support and adequate memory. Preparation itself uses only the CPU:

```powershell
python experiments/social_closed_loop/prepare.py run-folder --manifest artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json
python -m pip install -r run-folder/requirements-gpu.txt
python run-folder/run_gpu.py --root run-folder
```

The script does not allocate resources, authenticate to Colab, or select accounts. Keep service credentials and connection state outside the run folder. A production launcher should retain failure output as well as successful runs, bound execution time, and release its allocation after exporting and verifying evidence.

For the hash-verified export layout used in this study, run:

```powershell
python experiments/social_closed_loop/analyze.py downloaded-export analysis-folder
```

The analysis independently reconstructs the posterior targets, actions and Decimal body events. It reports probabilities against both teaching targets and the synthetic partner rule, separates same-content paraphrases from new-content probes, and retains all individual lives. Longitudinal contrasts use the common first held-out question across snapshots. It separately reports endpoint error across all six held-out contents when every endpoint includes them; otherwise the common-content comparison avoids mixing different question sets after an early death. Death is absorbing; 60 encounters is an observation cap, not a full natural-lifetime measurement.

Body transplants are read-only analyses of a saved learner. They do not reset the body of the continuing life. Lack of action changes across ordinary health states is a result to report, not an implementation test to force to pass.

The repository's CPU workflow checks the published file manifests, safely unpacks the GPU export, reruns the independent mathematical checks, reconstructs all 2,400 records, and compares the aggregate values with the published report data. It needs neither a GPU nor model downloads:

```powershell
python experiments/social_closed_loop/verify.py
```
