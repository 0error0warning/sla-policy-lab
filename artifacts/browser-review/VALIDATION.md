# Manual Chromium visual and interaction review

Verified 2026-11-10 UTC against the local demo served at `http://127.0.0.1:8768`. This is a real browser review, separate from the DOM/event harness. No GPU, model download, API provider, credentials or public deployment was used.

## Result

The inspected desktop and narrow layouts have no blocking overlap, broken chart, or page-wide horizontal overflow visible in the reviewed screens. The principal experiment and replay interactions worked. One cosmetic defect was fixed: the run button's diagonal-arrow character rendered as a missing-glyph square in this Linux browser. It is now a small decorative inline SVG; the label, action and accessibility name are unchanged. The corrected icon and a completed post-fix run were visually rechecked in Chromium.

## Review surface and limits

- Linux Chromium native window, desktop screenshot size 1364 × 1024. The desktop browser content area was approximately 1180 × 758 physical pixels at 100% zoom.
- Narrow review used a 500-pixel-wide browser content area at 125% browser zoom: approximately 400 CSS pixels wide, with the vertical scrollbar reducing available content width. This exercises the actual responsive CSS in Chromium, but is **not** a phone emulator, touchscreen test, real iOS/Android test, or exact 390 × 844 viewport test.
- DevTools was unavailable in the review environment. No console inspection, DOM measurements or zero-JavaScript-exception claim is made from this manual pass.
- Browser downloads visibly reached “Complete”; their downloaded bytes were not opened in this manual pass. JSON data and export mechanics were separately covered by the automated unit/API and DOM/event checks.
- The optional Playwright browser smoke script was not run. Network failure injection, malformed responses, reduced motion and alpha-zero trajectory equality remain automated non-visual checks, rather than claimed manual browser coverage.

## Observed checks

- Fresh reload completed the default offline 24-seed comparison and displayed four method cards, five plotted policies, replay controls, paired charts and complete metrics.
- Desktop replay slider selection, next-step button (turn 10 → 11), previous/restart controls, play and pause worked. Playback advanced from turn 1 to turn 13 before being paused.
- Switching from Seed 100 to Seed 101 reset the replay to its first turn and refreshed the displayed trajectories.
- Narrow layouts stacked the configuration and dashboard, kept parameter controls and mechanism cards in two columns, and changed decision detail panels to a single column. Headers, explanatory text and chart legends remained inside their panels.
- Selecting online mode and four paired seeds produced a completed run. Small Policy showed average 37.8 parameter updates and parameter delta 0.2977. LoRA showed average 39.3 updates and delta 0.0060. These are this run's synthetic results, not a claim of improved performance.
- Clicking the Toy LoRA legend and its timeline cell selected LoRA and turn 10. The corresponding decision panel showed cumulative 10 updates, per-step delta 0.000464, cumulative delta 0.003440, reward/gradient evidence, unchanged base and before/after hashes.
- Disabling online learning and rerunning yielded Small Policy 39.3 average lifespan, zero updates and zero parameter delta, as expected for the frozen control. Export triggered a completed download of about 1,949 KB.
- Further reload and repeated run-button use completed normally and displayed replay turn 1. No stuck loading state was observed.
- Complete metrics expanded on desktop and narrow screens. Horizontal scrolling moved the narrow table to its rightmost columns while the page stayed at the same horizontal position. The action timeline also had its own horizontal scrollbar.
- Browser zoom and width were restored to the original desktop view after review.

## Evidence publication

Twelve full-desktop screenshots were captured during the review and retained locally. They are not published because they include surrounding workspace context. This public report records the observations and limitations, with the automated check logs below.

## Automated checks rerun locally

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: 66 tests passed. [Log](tests.txt).
- `python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests`: passed.
- `node --check web/app.js`: passed.
- `python tests/dashboard/run_checks.py --repo .`: passed with five fresh simulator exports, including online and learning-disabled controls. [Log](dashboard.txt). This harness does not test browser rendering.

These are local checks only. No remote commit, push, CI status or deployment was verified by this review.
