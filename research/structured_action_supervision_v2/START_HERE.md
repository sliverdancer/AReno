# START HERE — SAS Tool-Readiness Bridge

Protocol: `SAS-TR-v2.0`

Branch: `research/structured-action-supervision-v2-tool-readiness`

Parent evidence: consumed `SAS-P0-v1.1`

Status: `B2_INVALID_INFRASTRUCTURE_KV_CACHE_OOM`

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
2. Treat `stages/B2/stage_result.json` as terminal for `SAS-TR-v2.0`.
3. Do not interpret the 64 HTTP failures as model/tool-call failures. The
   serving worker exhausted memory before returning any raw model response.
4. Do not modify and rerun the consumed B2 attempt. Any capacity repair must
   use the separately frozen `SAS-TR-v2.1` successor described in
   `SAS_TR_V2_1_SUCCESSOR_PLAN.md` and obtain new GPU authorization.
5. Keep validation, reserve, N128, and N512 unopened under v2.0.

## Evidence paths

- Consumed raw source:
  `../structured_action_supervision_v1/stages/Q1_v1_1/attempt_20260730/`
- Reproducible forensic analyzer: `analyze_v1_1_failure.py`
- Dataset and protocol freezer: `prepare_protocol.py`
- CPU contract test: `../../tests/test_sas_v2_tool_readiness_cpu.py`
- Scientific flow: `figures/tool_readiness_causal_flow.svg`
- Invalid B2 evidence: `stages/B2/attempt_20260801_invalid_infrastructure/`
- B2 closure: `stages/B2/stage_result.json` and `stages/B2/REPORT.md`

## Current decisions

- `PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION`
- `PASS_B1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`
- `KILL_SAS_STATIC_MASK_AS_PRIMARY_METHOD_NOVELTY`
- `KILL_CURRENT_MAIN_TRACK`
- `INVALID_B2_INFRASTRUCTURE_KV_CACHE_OOM`
- `SAS_TR_V2_0_TERMINAL_NO_SCIENTIFIC_RESULT`
- `SAS_TR_V2_1_SUCCESSOR_UNOPENED_GPU_AUTHORIZATION_REQUIRED`

## Claim boundary

The retrospective evidence is consistent with, but does not causally establish,
thinking-budget exhaustion. B2 is the causal factorial. Even a successful B2
only qualifies an inference interface; AF/LF performance, stability, and sample
efficiency remain unmeasured.
