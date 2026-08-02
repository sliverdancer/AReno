# RIST-v2 CPU-only instrument reconstruction

Branch: `research/rist-v2-instrument-reconstruction`

Status: `D2_TERMINAL_KILL_CURRENT_REVISION`

Parent terminal decision:
`../rist_v1_1/stages/P2_1/gpu_run_20260802/evidence/stage_result.json`

## Boundary

RIST-v1.1 is terminal and is not repaired, rerun, or spliced into this route.
Its complete Qwen cell is development-only evidence for diagnosing the failed
instrument. Gemma's incomplete cell is infrastructure evidence only. Neither
cell may contribute an observation to a future confirmatory analysis.

This route is CPU-only until a new protocol reaches an explicit GPU
authorization boundary. It must not start serving, inference, training,
checkpoint download or replacement, or held-out access.

## Current diagnosis and terminal result

The v1 analytic strata assumed a uniform policy over enumerated actions. That
construction measure did not transport to Qwen: all nine analytic-high tasks
had homogeneous success, while six analytic-low tasks produced mixed rewards.
The failure is therefore not merely Gemma memory pressure. The primary
construct, pre-training conditional reward resolution, must be estimated for
each checkpoint family on a development calibration pool and prospectively
validated on fresh nonces.

D0 reproduced this diagnosis and D1 validated the CPU uncertainty estimator.
D2 then returned `KILL_RIST_V2_D2_STRUCTURAL_POOL` because its frozen
byte-identical-regeneration gate evaluated false. Forensic analysis found that
the generated files are byte-identical, but the evaluator compares an in-memory
tuple with the JSON-reloaded list before checking bytes. This is a protocol
implementation defect, not scientific evidence. The consumed D2 result is not
repaired or rerun; the current revision is terminal.

## Required order

1. D0: freeze the v1.1 postmortem and evidence boundary.
2. D1: validate a model-conditional calibration estimator on CPU fixtures.
3. D2: terminal `KILL_RIST_V2_D2_STRUCTURAL_POOL`; stop this revision.
4. Any continuation requires a new protocol revision that fixes the evaluator
   before execution. It may reuse the design but not the consumed D2 result.
5. GPU work remains closed.

## Publication claim boundary

The surviving novelty is a prospective moderation study: whether a
pre-training, checkpoint-conditional reward-resolution measure predicts when
AF/LF/AN/LN supervision contrasts are gradient-resolvable under update-count
and trainable-token matching. Static masks, reward collapse, turn credit, and
tool-name/argument decomposition are not novel by themselves.

The route is currently a publishable question, not a publishable result. A
main-conference hook may upgrade only after independent training seeds show a
practically meaningful interaction with feasible uncertainty.

## Evidence roots

- prior terminal report: `../rist_v1_1/stages/P2_1/gpu_run_20260802/evidence/TERMINAL_REPORT.md`;
- prior novelty audit: `../rist_v1_1/stages/R0/FULL_TEXT_REVIEW.md`;
- D0 protocol and result: `stages/D0/PROTOCOL.md`, `stages/D0/stage_result.json`;
- D1 calibration contract: `stages/D1/PROTOCOL.md`;
- D2 terminal result and report: `stages/D2/stage_result.json`,
  `stages/D2/TERMINAL_REPORT.md`;
- route plan: `RESEARCH_PLAN.md`.
