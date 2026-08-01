# START HERE — SAS Tool-Readiness Bridge

Protocol: `SAS-TR-v2.1` (successful capacity-only successor to terminal v2.0)

Branch: `research/structured-action-supervision-v2-tool-readiness`

Parent evidence: consumed `SAS-P0-v1.1`

Status: `KILL_B3_CURRENT_GSPO_PILOT_STAY_DIAGNOSTIC`

## Answer first

The original AF-versus-LF research question remains scientifically interesting
but is not currently instantiated. Q1 v1.1 produced no valid tool trajectory,
so changing the loss mask could not change temporal credit.

This successor is a new instrument-qualification protocol. It tests whether
Qwen3's default thinking mode and a 128-token response ceiling prevented tool
calls from being emitted. It is not a retry of Q1 and cannot contribute efficacy
observations to Q1.

## First-turn behavior

1. Read `RESEARCH_PLAN.md`, `LITERATURE_REVIEW.md`, and
   `stages/B0/forensic_result.json`.
2. Treat `stages/B2/stage_result.json` as terminal for `SAS-TR-v2.0`; none of
   its request-error files enter the successor analysis.
3. Use `stages/B2_1/stage_result.json` as the current instrument decision.
4. The selected interface is Qwen3-0.6B with thinking disabled and a 128-token
   response budget (`N128`). Do not substitute D512 or N512 post hoc.
5. Read `stages/B3/stage_result.json` before proposing training. B3-A proved
   that all 16 task groups had zero within-group reward variation across eight
   samples. Factorial training is terminally closed under `SAS-B3-v3.0`.

## Evidence paths

- Consumed raw source:
  `../structured_action_supervision_v1/stages/Q1_v1_1/attempt_20260730/`
- Reproducible forensic analyzer: `analyze_v1_1_failure.py`
- Dataset and protocol freezer: `prepare_protocol.py`
- CPU contract test: `../../tests/test_sas_v2_tool_readiness_cpu.py`
- Scientific flow: `figures/tool_readiness_causal_flow.svg`
- Invalid B2 evidence: `stages/B2/attempt_20260801_invalid_infrastructure/`
- B2 closure: `stages/B2/stage_result.json` and `stages/B2/REPORT.md`
- Successful successor evidence: `stages/B2_1/attempt_20260801_pass/`
- Successor closure: `stages/B2_1/stage_result.json` and
  `stages/B2_1/REPORT.md`
- B3-A raw evidence: `stages/B3/sas-b3a-attempt-20260801-57ba60c/`
- B3-A closure: `stages/B3/stage_result.json` and `stages/B3/REPORT.md`

## Current decisions

- `PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION`
- `PASS_B1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`
- `KILL_SAS_STATIC_MASK_AS_PRIMARY_METHOD_NOVELTY`
- `KILL_CURRENT_MAIN_TRACK`
- `INVALID_B2_INFRASTRUCTURE_KV_CACHE_OOM`
- `SAS_TR_V2_0_TERMINAL_NO_SCIENTIFIC_RESULT`
- `PASS_B2_INTERFACE_N128`
- `STAY_DIAGNOSTIC`
- `KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL`
- `B3_FACTORIAL_TRAINING_NOT_OPENED`
- `STAY_DIAGNOSTIC`

## Claim boundary

The v2.1 factorial validates a tool-ready inference interface. B3-A then shows
that aggregate reward diversity hides zero within-task reward diversity under
the frozen GSPO sampling groups. Neither stage estimates AF/LF or full-call
versus name-only performance, stability, or sample efficiency. No
main-conference upgrade is supported.
