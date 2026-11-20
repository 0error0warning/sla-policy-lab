# Warm birth, unrestricted existing adapter: CPU development protocol

Locked before restoration and new outcomes, 2026-11-14. This is a bounded development experiment, not a confirmatory study. No GPU allocation, new model, thinking, new adapter, population refit, reward change, or hyperparameter search is permitted in this PR. Old negative results stay unchanged.

## Intervention and fixed machinery

Use the archived `consequence-policy` capture (batched, float32 exported Qwen prefixes), projection, and the original population adapters for seeds 63001–63004. Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` stays frozen. NumPy CPU scoring is a cached-prefix surrogate for the native vocabulary projection; this is not a new full-Qwen execution.

Four autonomous conditions: `prior` freezes `(W_A + W_B)/2`; `personal` starts at that mean and updates two partner scalars along `(W_A - W_B)/2`; `warmstart_adapter` starts at the same mean, removes the direction, and uses the existing `scratch` coefficient update; `scratch` starts at zero and uses exactly that same update branch. No change to health basis, partner routing, native action words, temperature 2, exploration 0.05, reference-life feedback, baseline -0.9, scratch step 0.7 or its normalization. The comparison warm/scratch isolates initialization within this implementation. Warm/personal also differs in optimizer and parameter space, not only prior direction. A mean can retain population knowledge even without supplying its contrast explicitly.

Health starts at 350, never resets or heals, and death is absorbing. Cap 256 encounters. Every eight encounters cross the two partners and two request frames twice; 24 layouts cycle in shuffled order. Action and maintenance uniforms and five bounded damage noises are paired across conditions by encounter, but each condition chooses actions and accumulates its own health and observations. No unchosen outcome or oracle enters learning. Tasks count answer actions, not independently generated answer correctness.

## Stage 0: restoration gate

Reconstruct all original pilot lives for `prior`, `personal`, and `scratch`: four archived seeds × four worlds × two replicas × three conditions = 96 lives. Use original batched cache and no refitting. Require exact trace lengths, sampled actions, responses and health; maximum probability error <= 2e-6 and scalar/normalized-feedback error <= 1e-8; scratch terminal weights within 2e-6. Verify source archive manifests and bind cache, projection, checkpoints, source and protocol hashes. Failure stops before new lives. These are recovery checks, not new independent evidence.

## Stage 1: fixed development matrix and budget

Pair archived birth seeds 63001–63004 with new environment seeds 78201–78204, respectively. Each has all four mapping worlds (00, 01, 10, 11) and two fixed task blocks; one stream per block. Total 32 lives per arm, 128 new autonomous lives, at most 32,768 encounters. These are four independent seed clusters; worlds, task blocks, prompts, layouts and turns are repeated measurements. No selection of worlds after results.

* `legacy`: live task 4 (seven minus two), wording 2, and held-out task 5, wording 2. This preserves the original task allocation. Its held-out endpoint changes task content, so it is not a paraphrase-only claim.
* `wording`: live task 0 (two plus two), wording 0, and held-out wording 1 of the identical task, frame, partner and layout. Both wordings appeared in population fitting, but only wording 0 is used in personal updates. This tests transfer of new personal experience between previously known expressions, not learning unseen language from scratch.

Use the original environment-tape algorithm, reindexing only task/wording for the second block. Named seed streams separate blocks; each block is identical across arms. CPU wall-clock cap for restore+run+analysis: 60 minutes; stop on invalid numerics or physical inconsistency, do not fill missing lives with replacements. No optional expansion in this PR even if the gate passes.

## Snapshots, same-history replay, and evaluator

Save birth, after 16/32/48 updates **only if still alive**, last preterminal parameters, and terminal parameters separately. Main endpoint is last preterminal: parameters actually used to make the final live choice, excluding its terminal feedback. Intermediate panels show retention and sample counts; do not treat survival-conditioned panels as independent trials. Record every injury followed by another actual decision, requests for maintenance, responses, update norms, action probabilities before/after each update at the original input, and number of nonterminal updates.

For every warmstart snapshot, replay its actual prefix from birth using (i) true normalized feedback, (ii) feedback permuted within partner × pre-action health bands [0,6), [6,15), [15,350], and (iii) a zero-born adapter on the same true history. Each recipient recomputes its own probabilities/gradients; no logged donor gradient is reused. True replay must recover warm weights. Shuffle only the observed prefix available at that checkpoint, so later/death feedback cannot enter an earlier probe; record the exact permutation, fixed points and group sizes. These are retrospective, off-policy sensitivity interventions without importance weighting, not autonomous lives or estimates of an alternative on-policy algorithm. No replay lifetime scores. No action labels or scalar-direction oracle are supplied to warm/scratch.

Probe both the live and held-out context sets over all 24 layouts, both frames/partners and original TEST_H=(2,3,4,6,10,18,90,300). Use identical inputs for birth/current/shadow. Report preferred answer-expression probability, answer participation, expression preference conditional on answering, partner-specific contrasts and both-partner success in mixed worlds. Evaluator-only expected total death risk uses the unchanged archived 32,768-draw physical risk estimator, fixed random stream and true partner rates. It never generates training labels. Report risk by health, not just answer suppression, and realized full-life turns/tasks separately.

## Locked development gates and stopping decisions

Evaluate each task block separately at last-preterminal held-out inputs. Thresholds below are engineering decisions, not significance tests or adequate sample-size claims. Show all four paired seed results, range and absolute levels.

1. Positive reference: original `personal` exceeds fixed `prior` by >= 10 percentage points in preferred answering at H=90/300, with >= 3/4 favorable seed means, in both blocks. If not, treat the development context/restore difference as unresolved and do not infer that a new method lacks capacity.
2. Warm learning: preferred answering exceeds prior by >= 10pp, same-history shuffled warm by >= 5pp, and autonomous scratch by >= 5pp; each contrast has >= 3/4 favorable seed means in each block. In worlds where the two partners actually prefer opposite expressions, both partners separately improve over prior in >= 6/8 lives per block. Conditional expression preference must also increase over prior in both blocks, to rule out a pure overall increase in answering. Mapping labels are not expression labels: because P17 and P42 have opposite base maps, worlds 00/11 have opposite expression preferences; 01/10 have equal preferences for the same frame. Report both types.
3. Retention: healthy H=90/300 answering decreases by at most 5pp from birth; total death risk at each of H=2/3/4/6 increases by at most 2pp. Require these on available held-out panels at 16/32/48 and last-preterminal, within each block, with counts disclosed.
4. Autonomous usefulness: mean tasks and turns do not decline versus paired fixed-prior lives in either block. Report actual learning opportunities and cap frequency.

All gates must pass before recommending new-seed/full-Qwen confirmation. A learning pass with retention failure directs the next question to preserving birth competence. If only the scalar-direction reference works, stop expansion and report the limit of this coefficient update and finite feedback. If neither works, investigate restoration/evaluation/context differences. No post-result changes to thresholds or selected seeds. The frozen risk-table route remains stopped.
