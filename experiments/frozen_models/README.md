# Frozen-model studies

This directory publishes the **standalone** 2026-11-10 L4 studies. The web app and its CPU teaching mechanisms retain their existing behavior. No pretrained weights or account credentials are stored in this repository, and no model was trained in these studies.

- [Current findings and limits](../../docs/MODEL_STUDIES.md)
- [Data, reports, and provenance](../../artifacts/model-studies/2026-11-10/README.md)

## Verify the saved evidence without a GPU

From the repository root, with the normal NumPy dependency installed:

```sh
python experiments/frozen_models/verify.py
```

This reads the frozen environment snapshot rather than a future version of the app. It verifies published file hashes, cross-checks numerical answer keys, recomputes scores, replays 88 model lives and 128 rule-policy lives, checks paired event tapes, and verifies the first-attempt/recovery diagnostic outputs. It does not install packages, download files, modify evidence, or allocate compute.

The archive also retains raw generated token IDs. An optional independent decoding check uses `tokenizers` and **already downloaded** official tokenizer files:

```sh
python experiments/frozen_models/verify.py --tokenizers-dir /path/to/tokenizers
```

That directory must contain `model_0`, `model_1`, and `model_2`, each with `tokenizer.json` and `tokenizer_config.json`. Pinned official source URLs and SHA-256 values are in [tokenizer-files.json](../../artifacts/model-studies/2026-11-10/tokenizer-files.json). No tokenizer or weight download occurs in the verifier. Model indices are 0 = Qwen3.5-4B, 1 = Qwen3-4B-Instruct-2507, 2 = Qwen3-1.7B.

The fixed rule controls can also be regenerated into a new file:

```sh
python experiments/frozen_models/baselines.py --output /tmp/sla-rule-baselines.json
```

The verifier regenerates these 128 lives and checks equality with the saved controls, including their chosen actions. The one-step rule considers only the currently visible load, protection and damage, with the declared reward and death cost; it does not use future events or the partner's true support probability.

## Prepare a reproduction

Preparation alone is CPU-only and copies only public inputs, the measured worker, and the frozen environment snapshot into a new directory. It does not copy answer keys or saved model outputs.

```sh
python experiments/frozen_models/run.py --study diagnostic-v2 --output /tmp/sla-diagnostic-inputs
```

Each run needs a fresh directory. The published evidence directory is protected against use as an output target. `--study frozen-v1` prepares the original three-model screen.

## Explicit GPU reproduction

The measured environment used a standard NVIDIA L4 with 23034 MiB, Python 3.13.16, torch 2.14.1+cu130, CUDA 13.0, and the versions in `requirements-gpu.txt`. Install a CUDA-enabled torch build for the target machine, then the listed dependencies. Differences from the recorded runtime must be reported; bitwise generation equality on another stack is not promised.

```sh
python -m pip install -r experiments/frozen_models/requirements-gpu.txt
python experiments/frozen_models/run.py --study diagnostic-v2 --output /content/sla-new-diagnostic --run
```

Only `--run` downloads pinned public weights and calls the model. The entrypoint invokes the original frozen `worker()` in separate processes with a fresh output root; it does not invoke the archived automatic dependency installer, allocate Colab hardware, authenticate accounts, or train models. Model 0 still loads the full Qwen3.5 ConditionalGeneration wrapper with text input. Missing `causal-conv1d` produces the recorded fallback; this is not an optimized throughput benchmark.

Run the experiment as actual foreground work in the notebook/CLI, and save outputs to durable storage during long jobs. The diagnostic study's first instance ended unexpectedly and its complete rollout files could not be recovered. The finished recovery used identical inputs, foreground execution, and host-side checkpoints. See the interruption record; do not pool attempts as extra independent samples.

Generation with `enable_thinking=True`, greedy decoding, and 1024 output tokens is a **separate combined condition** from non-thinking with 128 tokens. All eight measured thinking cases exhausted their token budget without a final channel. This is a budget-limited failure, not a general capability conclusion. Constrained JSON generation was reserved but was not executed.

The `executed/` files under the data archive are source provenance, not direct entrypoints. Their original filesystem assumptions and serialization behavior are retained so their hashes match the measured code.
