# RIST-v1.1 Successor Route

Status: `P2_1_TERMINAL_CLOSE_CURRENT_ROUTE`

Parent terminal result: `../../stages/P2/stage_result.json`

## Objective

Test whether conditional reward resolution prospectively predicts when
action-span supervision topology is gradient-resolvable in multi-turn tool
agent learning.

This successor does not reopen or splice `RIST-P2-v1.0`. The parent remains
terminal `INVALID_P2_INFRASTRUCTURE` with zero scientific trajectories.

## Required order

1. Complete the R0 full-text direct-substitute audit.
2. Apply the frozen R0 gate.
3. Only after R0 passes, prepare the E0 non-scientific canary contract.
4. Stop before starting any GPU process and request explicit authorization.
5. Never use qualification or held-out tasks for environment debugging.

## Current result

R0 returned `PASS_NOVELTY_CONDITIONAL_TO_E0`; its main-conference hook stayed
diagnostic. `RIST-E0-v1.1` is terminal `INVALID_E0_PREFLIGHT_STOP`: its frozen
top-level extension import named a module that the package does not register.
The actual extension import, CUDA, exact source, `areno check`, and empty GPU
process list passed, but no server or model request was started.

`RIST-E0-v1.2` minimally corrected that symbol and passed both model-family
canaries: each produced 8/8 raw responses, 8/8 parser-valid calls, and 8/8 exact
instructed tool/code pairs. Its decision is
`PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE`; it remains infrastructure-only
and does not upgrade the main-conference route.

`RIST-P2.1-v1.0` is terminal `INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`.
Qwen completed all 256 trajectories and passed every frozen per-model gate,
but produced at least two mixed-reward groups in only one reward stratum. The
frozen cross-family PASS condition therefore became unreachable even before
the second model completed. Gemma then exhausted GPU memory on the first task;
three HTTP-200 partial responses were visible in the server log but were not
retained by the transactional client, so its evidence cell is incomplete.

The route-management hook is
`CLOSE_CURRENT_RIST_V1_1_ROUTE_NO_UNCHANGED_RERUN_VALUE`. A Gemma-only
infrastructure repair cannot change the failed Qwen cross-stratum gate, so the
current RIST-v1.1 instrument is closed and P3 training must not open. This is
not a falsification of the broad research hypothesis: cross-family inference
is not estimable from this run, and Qwen remains diagnostic evidence only.

## Evidence roots

- R0 full-text decision: `stages/R0/stage_result.json`;
- R0 reviewer matrix: `stages/R0/reviewer_threat_matrix.json`;
- E0-v1.1 terminal result: `stages/E0/gpu_run_20260802/evidence/stage_result.json`;
- E0-v1.1 main-conference hook:
  `stages/E0/gpu_run_20260802/evidence/hook_result.json`;
- E0-v1.2 protocol and manifest: `stages/E0_v1_2/PROTOCOL.md`,
  `stages/E0_v1_2/EXECUTION_MANIFEST.json`;
- E0-v1.2 CPU freeze: `stages/E0_v1_2/cpu_freeze_result.json`;
- E0-v1.2 terminal result:
  `stages/E0_v1_2/gpu_run_20260802/evidence/stage_result.json`;
- E0-v1.2 independent audit:
  `stages/E0_v1_2/gpu_run_20260802/evidence/audit_result.json`;
- P2.1 protocol and manifest: `stages/P2_1/PROTOCOL.md`,
  `stages/P2_1/EXECUTION_MANIFEST.json`;
- P2.1 CPU freeze: `stages/P2_1/cpu_freeze_result.json`;
- P2.1 terminal result:
  `stages/P2_1/gpu_run_20260802/evidence/stage_result.json`;
- P2.1 independent audit:
  `stages/P2_1/gpu_run_20260802/evidence/audit_result.json`;
- P2.1 route-management hook:
  `stages/P2_1/gpu_run_20260802/evidence/hook_result.json`;
- P2.1 terminal report:
  `stages/P2_1/gpu_run_20260802/evidence/TERMINAL_REPORT.md`.

## Claim language

Prefer `gradient-resolvable` or `estimable under group-relative objectives`.
Do not claim formal statistical identifiability from low reward variation.
