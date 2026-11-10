# Dependency-free dashboard behavior checks

From the repository root:

```sh
python tests/dashboard/run_checks.py --repo .
```

Requires the existing Python/NumPy runtime and Node.js. No package or browser is installed. The runner generates five real simulator exports: default offline, support preset, single-seed zero-steering, online learning enabled, and online learning disabled. Generated files stay in the ignored `tests/dashboard/generated/` folder.

The Node harness executes the actual `web/app.js` against a small DOM/event simulator and checks rendering, request/fixture consistency, playback, pause, steps, speed, seed reset, selected policy, terminal states, focus retention, presets, dirty state, repeated runs, failure recovery, malformed results, JSON export, reduced-motion handling, single-seed intervals, online update evidence and learning-disabled controls.

This is **not a browser or layout engine**. It does not validate CSS, responsive layout, screenshots, native browser events or HTTP transport. The Python API suite covers local HTTP; `tests/ui_smoke.py` remains the optional real-browser test. Do not describe this harness as visual QA.
