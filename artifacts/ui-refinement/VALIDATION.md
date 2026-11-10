# UI refinement and verification

Verified 2026-11-10 against the local CPU demo. The simulator, model training, scoring and API behavior were not changed.

## Interface changes

- Warm off-white canvas, white panels and graphite text, with one blue action color. Muted teal, blue, ochre and violet identify mechanisms consistently across cards, charts and legends.
- Larger key values and labels. Shorter headings and fewer always-visible explanatory paragraphs.
- Common settings remain visible. Crack, protection duration, steering and seeds move into a native keyboard-accessible advanced-settings disclosure. Online-only learning controls appear only in online mode.
- Replay instructions, per-turn decision evidence, metric caveats and model details are available on demand. The non-LLM / uncalibrated teaching-setting boundary remains visible at the top. The boundary link opens the full disclosure, including repeated clicks.
- Invalid seed/count fields automatically reveal advanced settings. The loading, stale-result and error states remain explicit.
- Frozen online controls now say that their weights are frozen. A missing-glyph pause symbol was replaced by a plain text label. Narrow-screen chart labels are enlarged and the redundant cursor tag is hidden; the current turn remains visible in the toolbar.

## Real-browser review

Manual Linux Chromium review used the default desktop window (approximately 1180-pixel content width at 100% zoom) and a 500-pixel content width at 125% zoom (approximately 400 CSS pixels, minus the scrollbar). This is not a phone emulator or a real mobile-device test.

Observed on desktop and/or narrow screens:

- Default offline results, all four cards, five chart curves and five timeline rows render without visible page-wide overflow or overlapping controls in the inspected screens.
- Native disclosures open and close. The boundary link scrolls to and expands the full explanation.
- Switching to online mode reveals learning settings. Advanced controls remain reachable on the narrow layout.
- A four-seed online run completed. Small Policy showed 37.8 average updates and parameter delta 0.2977; LoRA showed 39.3 updates and delta 0.0060. These are synthetic results, not a performance-improvement claim.
- Play, pause, single-step, policy selection and sample selection worked; sample selection reset the timeline to turn one. LoRA's per-step probabilities, parameter changes and hashes remained available under per-turn details.
- Disabling updates and rerunning produced zero update counts and deltas. Export visibly completed a roughly 1,949 KB JSON download. Downloaded bytes were not opened during this visual pass.
- The final pause label and enlarged narrow chart axes were rechecked after their last edits. Browser width and zoom were restored afterward.

The optional Playwright smoke script was updated for the collapsed controls and structured result metadata, but was not executed. DevTools/console inspection, automatic DOM dimension assertions, touchscreen behavior and mobile OS behavior were not covered by this manual pass. Screenshots remain local rather than publishing surrounding browser/workspace context.

## Automated checks on final code

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: 66 tests passed. [Log](tests.txt).
- `python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests`: passed.
- `node --check web/app.js`: passed.
- `python tests/dashboard/run_checks.py --repo .`: passed against five fresh simulator exports. [Log](dashboard.txt).

The dashboard harness covers default/repeated runs, malformed responses, failure recovery, playback, export, reduced motion, zero-steering equivalence, online learning and frozen controls. It now also covers conditional controls, repeated boundary opening, invalid-field reveal and correct frozen labels. It is not a browser or layout engine.
