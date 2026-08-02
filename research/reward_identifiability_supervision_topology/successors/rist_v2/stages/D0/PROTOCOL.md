# RIST-v2 D0 postmortem protocol

Protocol: `RIST-D0-v2.0`

Mode: CPU-only, retrospective development diagnosis

## Inputs

- the immutable v1 P1 qualification task file;
- the complete v1.1 Qwen P2.1 result;
- the v1.1 terminal report and audit as provenance.

The Qwen cell is reclassified as development-only for v2. It may motivate and
test analysis code but may never be counted in a future qualification,
training, held-out, or confirmatory estimate. The incomplete Gemma cell is not
analyzed scientifically.

## Measures

- empirical mixed-group incidence by v1 analytic stratum;
- analytic mixed-probability Brier score against empirical mixed incidence;
- rank association between analytic and empirical task success/resolution;
- exact list of mixed structural cells;
- raw trajectory and task-hash integrity.

## Gate

Return `PASS_D0_MODEL_CONDITIONAL_MISCALIBRATION_TO_D1_CPU_CALIBRATOR` only if:

1. the task file hash matches the Qwen result;
2. all 32 tasks have exactly eight retained trajectories;
3. all 1,024 raw responses are retained;
4. no infrastructure error, retry, repair, fabrication, training, or held-out
   access is present;
5. the v1 analytic-high stratum does not realize a higher empirical mixed rate
   than analytic-low.

This PASS validates a failure diagnosis, not the scientific hypothesis.
