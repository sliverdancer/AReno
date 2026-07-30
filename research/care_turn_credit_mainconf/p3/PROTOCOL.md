# P3 bounded executability protocol

Protocol: `CARE-P3-PILOT-v0.1`
Freeze date: `2026-07-30`
Evidence class: qualification only
GPU state: `NOT_AUTHORIZED`

## Question

Can the real agentic GRPO path execute signed, calibrated turn-credit routing
for one bounded update while preserving exact action traces, fixed-budget
normalization, matched audit cost, and complete structured artifacts?

This stage does not test whether CARe improves learning. With one optimizer
step, the logged rollout reward is measured before the resulting update and
cannot be interpreted as post-training performance.

## Treatments

| Field | CARe | Uncalibrated control |
|---|---|---|
| Behavior policy | Same frozen checkpoint and seed | Same |
| Rollout task/batch | Same controlled task, 20 trajectories | Same |
| Exact calibration audits | Same 10 even prompt blocks | Same |
| Cheap signed scorer | Same deterministic scorer | Same |
| Confidence budget prefix | Same, no backfill | Same |
| Threshold | Apply calibrated threshold | Ignore threshold |
| Update blocks | 10 odd prompt blocks | Same |
| Token budget | 256 per update trajectory | Same |

The two arms deliberately pay the same audit cost. A difference in the pilot
can therefore be attributed to calibrated abstention rather than receiving
more counterfactual labels.

## Leakage and dependence controls

Calibration and update trajectories come from one frozen behavior-policy
batch, but are disjoint prompt blocks. Calibration rows receive budget zero and
are fully masked; their oracle labels cannot enter the optimizer update.
`n_samples = 1` and unique prompt indices make the trajectory the calibration
unit. Dependent turns inside one trajectory are never counted as independent
calibration observations.

Invalid calibration trajectories are retained as pessimistic maximum-mass
wrong-sign blocks. Invalid update trajectories remain in the strict validity
denominator and receive a full-abstention update; no missing action is
synthesized.

## Frozen workload

- Checkpoint: ModelScope `Qwen/Qwen3-0.6B`, downloaded from `master` and
  accepted only if every file matches `modelscope_asset.json`.
- Task seed/count: `7301` / 64 deterministic unique records.
- Arms: `care`, `uncalibrated`.
- Training seeds: `3101`, `3102`, `3103`.
- Steps: one optimizer step per run.
- Batch: 20 trajectories, `n_samples = 1`, mini-batch 5, gradient accumulation
  4.
- Generation: five calls per trajectory, 64 new tokens per call, thinking
  disabled.
- Expected model calls: 600 across six runs.
- Planned exact terminal evaluations: 300 per run when every calibration
  trajectory is valid.
- Proposed host: one AutoDL A800 80GB, publicly listed at CNY 4.98/hour when
  checked on `2026-07-30`.
- Compute ceiling: 60 minutes per run; six training GPU-hours; eight total
  billed instance-hours including setup; CNY 60 total including storage.

At the checked rate, eight instance-hours cost CNY 39.84 before storage. The
CNY 60 ceiling leaves a small buffer. Stop before provisioning if the live rate
or required storage would exceed that cap. The one-hour run limit is a hard
kill threshold, not a runtime prediction. No GPU memory-fit or wall-clock claim
has been verified locally.

## Qualification estimands

No hypothesis test is performed with three qualification seeds. The required
descriptive checks are:

1. strict valid-trajectory rate;
2. per-step reward mean and standard deviation;
3. `trainable_tokens` and `masked_response_tokens`;
4. selected and masked routed mass;
5. audit calls and calibration/update block counts;
6. calibrated threshold;
7. command exit status and single-GPU wall time.

## Mechanical gate

Return `PASS_P3_EXECUTABILITY_TO_P4` only if all conditions hold:

1. all six frozen commands finish with exit code zero within 60 minutes each;
2. all six metric rows and raw logs exist and are aligned to the manifest;
3. every run has exactly 10 calibration and 10 update blocks;
4. strict valid-trajectory rate is at least 95% in each arm when pooled over
   its three seeds;
5. at least two of three runs per arm have reward standard deviation above
   zero, and the arm-level mean of run reward means lies strictly between 0.05
   and 0.95;
6. all required values are finite and no log records a non-finite optimizer
   event;
7. paired arms have identical audit calls for every seed;
8. CARe has positive selected/trainable mass but strictly less selected mass
   than the uncalibrated arm in at least two of three paired seeds.

Return:

- `KILL_P3_EXECUTABILITY` for crashes, timeouts, missing artifacts, non-finite
  updates, or validity below 95%;
- `KILL_P3_SIGNAL_DEGENERATE` for collapsed reward or zero CARe signal;
- `KILL_INCREMENTAL_PILOT` if calibrated abstention never changes selection;
- `INVALID_P3_PROTOCOL` for source/asset/manifest drift or unmatched audit cost.

No failed seed may be silently removed. A hardware failure may be rerun only
under the frozen seed after retaining the original failure record and before
any outcome-dependent protocol change.

## Claim boundary

A pass supports only:

> The frozen CARe path executed on the recorded controlled qualification task
> and emitted complete artifacts.

It does not support improved learning, real-task transfer, a new semantic
scorer, lower FLOPs, or a main-conference acceptance claim.
