# RIST-v1.1 Successor Route

Status: `P2_1_CPU_READY_AWAITING_GPU_AUTHORIZATION`

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

`RIST-P2.1-v1.0` is now CPU-frozen. It preserves the 32 qualification tasks,
eight rollout seeds, sequential request order, two checkpoint families, and
all scientific gates from the unconsumed P2-v1.0 client while requiring the
E0-proven native stack. E0 throughput forecasts about 3.65 GPU-hours for 2,048
requests, so the old two-hour estimate is infeasible and the new hard ceiling
is five GPU-hours. No qualification request or training has been executed.

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
- P2.1 CPU freeze: `stages/P2_1/cpu_freeze_result.json`.

## Claim language

Prefer `gradient-resolvable` or `estimable under group-relative objectives`.
Do not claim formal statistical identifiability from low reward variation.
