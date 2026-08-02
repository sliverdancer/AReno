# RIST-v2 research plan

Date: `2026-08-02`

Status: `D2_TERMINAL_KILL_CURRENT_REVISION`

Terminal note: D0 and D1 passed their CPU diagnostic/engineering gates. D2
returned `KILL_RIST_V2_D2_STRUCTURAL_POOL` because the frozen evaluator treated
tuple-versus-list representation drift as failed reproducibility. Generated
files were independently found byte-identical, but the consumed result is not
repaired or rerun. This is no evidence against H1-H4; it closes this protocol
revision before any model, GPU, training, or held-out access.

## Research question

Under group-relative optimization of multi-turn tool agents, does
checkpoint-conditional pre-training reward resolution prospectively moderate
the causal effect of temporal action coverage and call-content eligibility on
strict task success, stability, and sample efficiency?

## Competing hypotheses

### H1 — Checkpoint-conditional resolution

The v1 failure arose because uniform combinatorial difficulty is not a valid
proxy for a language model's action distribution. A model-conditional
calibration measure on development nonces will predict mixed-reward groups on
fresh nonces from the same structural cell.

Prediction: prospectively calibrated cell-level mixed-group probability has a
positive rank association with fresh-nonce mixed-group incidence in both model
families. Falsifier: the association is non-positive or the selected cells do
not retain their declared ordering on the independent qualification pool.

### H2 — Saturation, not lack of action diversity

Qwen's homogeneous groups in v1 were largely success saturation: structurally
different action sequences existed, but the model deterministically recovered
the oracle. Controlled ambiguity and dependency must change model success
probability without changing the four-turn causal skeleton.

Prediction: calibrated structural cells span collapsed, transition, and
resolved reward regimes while preserving executable four-turn behavior.
Falsifier: added ambiguity causes parser/interface collapse rather than graded
strict-reward variation.

### H3 — Resolution is family-specific but transportable by cell

The same exact task need not have the same difficulty for Qwen and Gemma, but
structural cells selected using independent calibration nonces can preserve an
ordering within each family.

Prediction: a hierarchical model with checkpoint-specific intercepts and
shared structural slopes predicts fresh-nonce reward resolution better than
the v1 uniform-policy score. Falsifier: cell rankings are unstable across
nonces or no common set provides two signal-bearing bands in both families.

### H4 — Token mass explains apparent topology effects

Any later all-action advantage may be caused by more trainable tokens rather
than causal coverage of early actions.

Prediction: step-matched AF/LF or AN/LN contrasts shrink materially under
cumulative-trainable-token matching. Falsifier: the resolution-by-topology
interaction survives both matching schemes with a practically meaningful
effect and seed-level uncertainty excluding the frozen null region.

## Stages and mechanical gates

### D0 — Terminal-evidence postmortem (CPU)

Use only the complete v1.1 Qwen qualification cell as development evidence.
Quantify analytic-to-empirical miscalibration and freeze the no-splicing
boundary. Do not read held-out.

Gate: proceed only if inputs are hash-consistent, all 256 Qwen trajectories are
retained, and the mismatch can be reproduced without modifying v1 artifacts.

### D1 — Calibration estimator (CPU)

Implement task-level binomial uncertainty, group-size-specific mixed-reward
probability, and fail-closed cell summaries. Unit tests use synthetic records
plus the archived Qwen development cell.

Gate: deterministic tests pass; no model or task claim is made from synthetic
fixtures.

### D2 — Candidate-pool and split freeze (CPU)

Generate structural cells with multiple independent nonces for:

- calibration-development;
- prospective qualification;
- sealed held-out.

Selection operates at the structural-cell level, never by retaining individual
tasks with convenient outcomes. Exact qualification and held-out outcomes must
not be used to change the generator or thresholds.

Gate: reproducible, signature-disjoint splits; unique strict oracle; sufficient
cell replication; no held-out read by selection code.

### E0 — Representative serving canary (future GPU authorization)

Use independent canaries matching the maximum prompt length, four-turn KV
cache growth, output limit, and sequential request settings of D2. Require
explicit memory headroom after the fourth turn and append-only response
journaling before scientific tasks open.

### P2 — Prospective cross-family qualification (future GPU authorization)

For each family, require:

- executable four-turn interface;
- at least one collapsed negative-control cell;
- at least two signal-bearing resolution bands confirmed on fresh nonces;
- monotone calibration-to-qualification ordering with uncertainty;
- complete append-only raw evidence and zero repair/retry.

P2 may validate the moderator instrument but cannot estimate training effects.

### P3 — Paired factorial training pilot (future separate authorization)

Run AF/LF/AN/LN with at least three paired training seeds. Report run-level
uncertainty, non-zero-advantage groups, effective trainable tokens, span-wise
gradient norms, first irrecoverable error, and name/argument accuracy. Run both
step-matched and cumulative-trainable-token-matched schedules.

### P4/P5 — Power freeze and confirmation

Use P3 run-level variance for prospective power. Confirm across at least two
model families, two algorithms, a synthetic control environment, and a real
multi-turn tool benchmark. Held-out opens once under the frozen analysis.

## Main-conference hook

- D0/D1/D2 pass: `STAY_DIAGNOSTIC`; no paper upgrade.
- E0 pass: infrastructure only; no paper upgrade.
- P2 pass: `STAY_DIAGNOSTIC_OPEN_P3`.
- P3 with a stable interaction and feasible powered design:
  `UPGRADE_MAIN_TRACK_CANDIDATE`.
- P5 with consistent interaction, token-matched robustness, real-environment
  validity, and reproducible artifacts: `GO_MAIN_TRACK`.

Any direct substitute, non-transporting calibration, infeasible independent
seed requirement, or terminal protocol violation closes the current route.
