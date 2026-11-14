# Frozen decision-interface diagnosis

Read [the preregistered protocol](PROTOCOL.md). This is frozen inference, not
individual learning. It follows the complete direct and incomplete thinking
results in `experiments/frozen_capability` without modifying those artifacts.

```sh
python -m experiments.frozen_interface.design --out /tmp/interface-fixtures
python -m unittest tests.test_frozen_interface tests.test_frozen_capability -v
```

Only public prompts, fixed model manifests, generation configuration, and GPU
worker modules are uploaded. The scorer and answer key stay on CPU. The worker
tests final-channel completion, measured throughput, and long-context cache
before any scored generation. Raw outputs, failures and stopping reasons are
preserved. Final replay instructions and results are added after execution.
