# Existing-history CPU attribution, 2026-11-14

This is an exploratory follow-up to the already observed warmstart development result. Fix the following analysis before executing it; it is not a preregistered confirmation or a new autonomous experiment. No GPU, model changes, new life/environment seeds, tuning, reward changes, or body changes. Preserve the #28 artifacts. Use all 32 archived warmstart histories and, for descriptive comparison, all 32 archived scalar-direction personal histories. Four original seed clusters remain the unit of reporting.

## Two questions

1. Which actual experiences contributed to the increase in partner-appropriate answering?
2. Are there few usable low-health experiences, and do updates from the experiences that exist reach low-health decisions?

The two mechanisms can coexist. Existing histories cannot identify what would happen after collecting a new low-health curriculum; do not infer that more experience must fix the method.

## Measurements and cutoffs

Recover each actor from its archived birth and recompute each chosen-action update, confirming recorded probabilities and last-live/terminal weights. Primary endpoint excludes the terminal update, as in #28. Terminal updates are measured separately and cannot count as same-life learning.

For every nonterminal update, measure changes in the fixed held-out grid used in #28: healthy preferred-answer probability and total answering at H=90/300, conditional expression preference, and expected death risk at H=2/3/4/6. Each per-update change is on identical inputs before/after the update. Sum these along the actual trajectory: the sum must equal final-minus-birth. This is an exact ordered path decomposition, not order-independent causal credit or a Shapley value.

Also evaluate the original request at its original health, the original request with only health replaced by 2 or 6, the matching held-out request at original health (only wording changes in the wording block; task also changes in legacy), and the actual next request/current health. Record the previous chosen action probability, total answering, and distribution TVD. Split next-request transfer by whether partner identity changes. The known partner router is provided architecture, not learned identity discovery.

Categories: answer+maintenance, answer+refusal, ask+maintenance, ask+refusal, silence. Subdivide answer events into actually favored/unfavored expressions using evaluator-only true rates. These labels never enter training. Group separately by pre-action health <=6, (6,15], (15,90], >90 and by advantage sign (normalized feedback + 0.9). Report counts, mean per-update effects, summed per-life effects and all four seed means. Maintenance and damage are coupled in this simulator; category attribution does not independently identify their biological/social meanings.

## Fixed-history omission interventions

For each warmstart prefix before terminal feedback, replay from the same birth while omitting updates for (a) maintained answers, (b) refused answers, (c) all nonanswers, and separately each of the four health bands. Preserve the recorded actions, observations, physical states, order and original feedback for retained events. Recompute each recipient's current gradient. No resampling actions or new settlements. Compare final probes against normal replay; no lifetime/task scores for these interventions.

For the three action/response omissions, add five predetermined count-matched masks, sampling omitted positions within the same partner × four health bands. They preserve omitted-update counts in those strata, not reward distributions; report overlap and available strata. These are within-history mask replicates, not new independent seeds. Report named omission minus the mean matched omission and each mask result. Omission effects can interact and need not add up.

Replaying the existing scalar-direction personal rule on the same warmstart histories is an additional structural comparison: same birth, actions, feedback and update opportunities, recipient gradients. It is retrospective/off-policy without importance weighting, not evidence of that recipient's on-policy lifetime. It changes update space and algorithm, so it cannot by itself isolate the health basis as the only cause.

## Analytic transfer check

For a warmstart update, before float32 rounding,

`delta_W[p] = eta * g(h_source) outer x_source outer score_action`.

Its logit effect on a fixed probe of the same partner is

`delta_z = eta * dot(g(h_probe), g(h_source)) * dot(x_probe, x_source) * score_action`.

The actual parameter update has float32 rounding; log that residual. Validate the factorization numerically against a direct contraction. Report `dot(g(H),g(h_source))/dot(g(h_source),g(h_source))` at H=2/6, alongside actual probability/risk changes. This isolates the designed health overlap in the *linear score effect*; it is not a probability ratio or proof about all possible adapters. Compare parameter/head-logit movement with decision movement to separate a blocked channel from merely saturated softmax.

## Boundaries and stop

Maximum 60 CPU minutes; run the fixed analyses once. Stop on a source mismatch, nonfinite result or failed replay identity, fixing implementation errors transparently before re-execution. No significance tests or expansion decision from these reused four clusters. Outputs: per-update effects, per-life omissions and matched masks, health opportunity/overlap tables, reproducible summary, numerical audit and Chinese report. Strong terminal or same-history replay effects are diagnostic only. Do not modify the old result gate.
