# START HERE — SAS Tool-Readiness Bridge

Protocol: `SAS-TR-v2.0`

Branch: `research/structured-action-supervision-v2-tool-readiness`

Parent evidence: consumed `SAS-P0-v1.1`

Status: `B1_CPU_FROZEN_GPU_AUTHORIZATION_REQUIRED`

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
2. Verify `stages/B1/manifest.json` and run the CPU test before using a GPU.
3. Do not modify the four B2 cells, their sampling seeds, split membership, or
   selection rule after outcome inspection.
4. B2 is inference-only but requires GPU serving, so request explicit
   authorization under `AGENTS.md`.
5. Stop if neither `N128` nor `N512` passes. Do not escalate to a larger model,
   relax validation, or manufacture calls inside this protocol.
6. If a cell passes calibration, validate only the prespecified selected cell
   on the untouched validation split. The reserve split remains unopened.

## Evidence paths

- Consumed raw source:
  `../structured_action_supervision_v1/stages/Q1_v1_1/attempt_20260730/`
- Reproducible forensic analyzer: `analyze_v1_1_failure.py`
- Dataset and protocol freezer: `prepare_protocol.py`
- CPU contract test: `../../tests/test_sas_v2_tool_readiness_cpu.py`
- Scientific flow: `figures/tool_readiness_causal_flow.svg`

## Current decisions

- `PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION`
- `PASS_B1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`
- `KILL_SAS_STATIC_MASK_AS_PRIMARY_METHOD_NOVELTY`
- `KILL_CURRENT_MAIN_TRACK`
- `B2_GPU_INFERENCE_UNOPENED`

## Claim boundary

The retrospective evidence is consistent with, but does not causally establish,
thinking-budget exhaustion. B2 is the causal factorial. Even a successful B2
only qualifies an inference interface; AF/LF performance, stability, and sample
efficiency remain unmeasured.
