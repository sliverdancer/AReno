# Competing Hypotheses, Predictions, and Falsifiers

Protocol: `CARE-P0-v0.1`
All numerical thresholds below must be frozen before confirmatory outcomes are
observed.

## Core causal question

When a turn-level credit estimator is noisy, does abstaining on turns whose
credit sign is not reliable improve agent learning after matching environment
interactions and supervision-token density?

## H1 — Wrong-sign contamination is the operative mechanism

Uniform or uncalibrated fine-grained credit reinforces harmful turns in
successful trajectories and punishes useful turns in failed trajectories.

Predictions:

- CARe reduces audited wrong-sign gradient-mass risk to the declared level.
- It improves held-out success and reward AUC relative to the same base scorer
  without abstention.
- Gain grows with horizon and estimator noise.

Falsifiers:

- audited sign risk is already low for the uncalibrated scorer;
- CARe misses its risk target;
- or better calibration does not translate to learning improvement.

## H2 — Density regularization, not attribution, explains any gain

Masking fewer turns may simply regularize optimization, independent of whether
the chosen turns deserve credit.

Predictions:

- random-turn and magnitude-only masks matched on trainable tokens perform as
  well as CARe;
- causal/sign accuracy of selected turns does not predict downstream gain.

Falsifier:

- CARe beats all density-matched controls with a confidence interval excluding
  the preregistered negligible-effect region.

This is the main alternative explanation and must be tested, not discussed away.

## H3 — Credit magnitude is sufficient; formal risk control adds no value

Ranking turns by `abs(credit)` or a judge confidence score may capture all useful
information. Conformal calibration and abstention may be unnecessary overhead.

Predictions:

- top-k magnitude routing matches CARe at the same token and interaction budget;
- nominal risk levels do not track empirical sign error.

Falsifier:

- CARe achieves both lower sign error and better held-out performance than
  top-k routing, especially under injected or naturally occurring scorer noise.

## H4 — Counterfactual auditing costs more than it saves

Sparse counterfactual labels, environment snapshots, and calibration may consume
so many extra environment calls that sample-efficiency gains disappear.

Predictions:

- CARe wins per optimizer step but loses per total environment transition,
  GPU-hour, or wall-clock budget;
- more ordinary rollouts outperform fewer risk-calibrated updates.

Falsifier:

- CARe remains better on the primary interaction-matched analysis and is
  non-inferior on wall-clock/GPU-hour sensitivity analyses.

## H5 — On-policy drift invalidates calibration

A threshold calibrated under one behavior policy may not control risk after the
policy changes.

Predictions:

- empirical sign risk rises between calibration refreshes;
- risk increases with policy KL and task-distribution shift;
- a frozen threshold fails while per-iteration block recalibration succeeds.

Falsifier:

- trajectory-block calibration at the frozen refresh schedule meets the risk
  target across policy iterations and held-out task strata.

## Formal working target

For trajectory `tau` with turns `t = 1..T`, let `c_t` be the latent
counterfactual credit of the chosen action and `c_hat_t` a cheap estimator. CARe
learns or computes a score `q_t` for wrong-sign risk. Before calibration, a
deterministic confidence-ranked prefix is capped at gradient-token budget `B`.
A trajectory-block calibration set chooses a threshold `lambda` such that
expected wrong-sign gradient mass, normalized by the fixed budget `B`, is
controlled at target `alpha` under the frozen exchangeability assumptions.

The wrong-sign fraction conditional on selection is not the guaranteed target:
that ratio is generally non-monotone in `lambda`. It remains a mandatory
diagnostic.

The routed weight is:

```text
w_t = sign(c_hat_t) * magnitude(c_hat_t)
      if q_t <= lambda and t is admitted by the token budget
      else 0
```

The implementation may use soft weights, but the primary method must preserve
an explicit abstention state and signed per-turn credit. Static issue #199 masks
alone cannot represent this target.

## Primary confirmatory success gate

The route is scientifically promising only if all are true:

1. **Risk validity:** empirical block-level wrong-sign gradient-mass risk is at most
   `alpha + 0.03` for target `alpha = 0.10`, with the preregistered upper
   confidence bound satisfying the protocol.
2. **Learning:** the pooled task-family effect of CARe versus the identical base
   scorer without abstention has a 95% hierarchical-bootstrap interval above
   zero and CARe is positive on at least two of three primary task families.
3. **Practical effect:** at least two task families improve by the frozen
   practically meaningful threshold, initially proposed as 5 absolute success
   points or an equivalent normalized-return threshold.
4. **Matched controls:** CARe beats random-turn and magnitude-only routing under
   both interaction-matched and trainable-token-matched analyses.
5. **Cost:** gains do not reverse when all counterfactual audit calls are
   included in the environment-interaction denominator.

If risk control succeeds but learning does not, the result is a valid negative
method study, not a positive method paper.

## Gate decisions

| Gate | Pass | Kill / block |
|---|---|---|
| P0 novelty | No direct substitute; residual gap survives full text/code | Direct substitute or merely incremental wrapper |
| P1 theory/oracle | Block-risk derivation and exact synthetic oracle agree | Guarantee requires violated assumptions or oracle is ambiguous |
| P2 instrument | Per-turn offsets, signs, budgets, and zero-update masks pass CPU tests | Fabricated calls, collapsed turn boundaries, seed gaps |
| P3 pilot | Executability >=95%, non-degenerate rewards, complete artifacts | Parser failure, reward collapse, non-finite update, missing provenance |
| P4 resource | Powered design fits approved ceiling | `KILL_UNDERPOWERED_OR_UNAFFORDABLE` |
| P5 confirmatory | All five primary conditions above | Null, reversal, invalidity, or uncontrolled multiplicity |
| P6 transfer | Same frozen method transfers without task-specific retuning | Positive result depends on one oracle/task family |
