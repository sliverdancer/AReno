# P3-v0.2 frozen GPU gate decision

Protocol: `CARE-P3-PILOT-v0.2`

Execution date: `2026-08-01`

Provider: AutoDL

Hardware: one NVIDIA A800 80GB PCIe

Frozen source commit: `c96bcf2da464dff36593d43c8d29991d4b998059`

Decision: `KILL_P3_SIGNAL_DEGENERATE`

Downstream state: `P4_P5_P6_P7_UNOPENED`

## Answer first

All six frozen commands completed successfully and emitted aligned artifacts,
but both arms had collapsed qualification reward in every seed. The unchanged
P3 gate therefore returns `KILL_P3_SIGNAL_DEGENERATE`. The executable routing
path and nonzero CARe selection do not override the prespecified reward gate.

No run was removed, repaired, or selectively rerun. P4 is not opened.

## Execution and resource evidence

| Run | Exit | Wall seconds | Reward mean | Reward std | Selected tokens | Masked tokens |
|---|---:|---:|---:|---:|---:|---:|
| care-3101 | 0 | 62.271 | 0.0 | 0.0 | 420 | 1,481 |
| uncalibrated-3101 | 0 | 58.846 | 0.0 | 0.0 | 800 | 1,101 |
| care-3102 | 0 | 58.598 | 0.0 | 0.0 | 477 | 1,413 |
| uncalibrated-3102 | 0 | 58.662 | 0.0 | 0.0 | 797 | 1,093 |
| care-3103 | 0 | 58.831 | 0.0 | 0.0 | 437 | 1,421 |
| uncalibrated-3103 | 0 | 59.409 | 0.0 | 0.0 | 797 | 1,061 |

The six commands used `0.09906004493538705` single-GPU hours in total. Every
run stayed below the 60-minute per-run ceiling and the aggregate stayed below
the authorized six-GPU-hour ceiling.

## Mechanical gate audit

| Frozen condition | Evidence | Result |
|---|---|---|
| Six commands exit zero within 60 minutes | 6/6 completed; maximum 62.271 seconds | Pass |
| Six aligned metric rows and raw logs | Six unique `(arm, seed, step)` rows; full archive retained | Pass |
| 10 calibration and 10 update blocks per run | 10/10 in all six rows | Pass |
| At least 95% strict validity per arm | 60/60 valid trajectories in each arm | Pass |
| Non-degenerate reward | All six means and standard deviations are exactly 0.0 | **Fail** |
| Finite required values and optimizer statistics | Collector accepted all rows; parsed train statistics contain no non-finite numeric values | Pass |
| Matched audit calls | 300 calls in both paired arms for every seed | Pass |
| Positive but reduced CARe mass | 420<800, 477<797, and 437<797 | Pass |

The failing reward condition maps directly to
`KILL_P3_SIGNAL_DEGENERATE` in the frozen v0.1 gate inherited by v0.2. CARe's
selected and masked mass is nonzero, so this is not a zero-routing failure.

## Data-quality and lineage notes

- `pilot_steps.csv` contains exactly six unique arm/seed/step rows and all
  required fields.
- `pilot_steps.json` contains the same rows and embeds the frozen manifest.
- The collector rejected missing tags, misaligned steps, duplicate events,
  non-finite values, and incomplete arm/seed sets before producing the files.
- The full compressed archive preserves TensorBoard events, per-run logs,
  rollout samples, turn-credit diagnostics, manifest, dataset, and executor
  results.
- `GPU_EXECUTION_NOT_AUTHORIZED` is the intentionally preserved CPU-preparation
  marker. The later explicit authorization is recorded by
  `gpu_authorized_execution_started_at.txt`, the six completed run records,
  and this decision; the stale marker must not be read as the final run state.

## Evidence

Evidence root:

`research/care_turn_credit_mainconf/p3_v02/evidence/autodl_a800_20260801_kill_p3_signal_degenerate/`

Key SHA-256 values:

- dataset: `bdb6fd0f502a9f7a7bb2e2947a09c89d884ffada7364fb92c81d4105c38fa03d`;
- manifest: `ba096f0890bf5d7c601654de64c8485fc2c1b28f896a3fc4a31e58c81ece84ff`;
- run results: `38f2c123533b78e4ae787a1860f8b9c37e1de8bd0a9652f70252194cce95ceab`;
- CSV: `ca44ca5e8118c3905be60c41b278a3619d73a698da4966ab3a7202851cf1911f`;
- JSON: `e832af71f46a19718676d16a1659b85d818ba5880ae41eff2f19e311cb5418cf`;
- full archive: `f51d6131a9eddbb858b257f8a7666ca1ae2efb91d482413d54e1102e701b8357`.

## Claim boundary

This outcome supports only that the frozen CARe path executed, routed nonzero
token mass, and emitted complete artifacts on the recorded controlled task. It
does not support efficacy, calibration validity, transfer, sample-efficiency,
compute-saving, or main-conference method claims. The reward collapse makes
this qualification evidence unusable for freezing P4 power or resource
parameters.

The consumed v0.2 protocol is terminal. Any scientifically different task,
reward, model, arm, estimator, or retry would require a new versioned research
decision and cannot be spliced into this result.
