# Reward Identifiability x Supervision Topology

Branch: `research/reward-identifiability-supervision-topology`

Protocol family: `RIST-v1`

Status: `RIST_V1_PARENT_INVALID__V1_1_SUCCESSOR_TERMINAL`

## Owner goal

Determine when action-span supervision effects in multi-turn tool-agent learning
are statistically identifiable, rather than assuming that a mask comparison is
meaningful under every task and reward distribution.

The target contribution is a causal diagnostic benchmark, not a new static
masking method. The primary interaction is:

- reward resolution / within-task outcome variation; crossed with
- temporal supervision breadth (`all actions` versus `last action`); and
- call-content retention (`full call` versus `tool name only`).

## Prior terminal result

`SAS-B3-v3.0` is closed with
`KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL`. Its 128 trajectories are
historical motivation only. They may not be repaired, selectively rerun, or
spliced into `RIST-v1`.

## Current terminal result

P0 returned `PASS_NOVELTY_CONDITIONAL` and P1 returned
`PASS_P1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`. P2 then returned
`INVALID_P2_INFRASTRUCTURE`: Qwen serving failed twice before producing any
scientific response, triggering the frozen repeated-server-failure stop rule.
No trajectory or training result exists, Gemma was not served, and the research
hypothesis remains unestimated. The main-conference hook is
`INVALID_PROTOCOL_STOP`.

The explicitly authorized `RIST-v1.1` successor is also terminal. Its P2.1
qualification completed the Qwen cell but failed the frozen cross-stratum
mixed-group gate; Gemma then encountered an out-of-memory infrastructure error
with incomplete retained evidence. The formal result is
`INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`, and the route hook is
`CLOSE_CURRENT_RIST_V1_1_ROUTE_NO_UNCHANGED_RERUN_VALUE`. No training or
held-out evaluation occurred, and P3 must not open. See
`successors/rist_v1_1/START_HERE.md` for the exact claim boundary.

## First-turn behavior

1. Read this file and `RESEARCH_PLAN.md`.
2. Inspect the latest stage result and its exact manifest.
3. Execute only the next unopened stage.
4. Apply the frozen gate mechanically.
5. Run the main-conference hook after every completed stage.
6. Stop at `KILL`, `INVALID`, a GPU authorization boundary, or the final
   confirmatory conclusion.

## Claim boundary

CPU enumeration and simulated-policy qualification can validate task
construction and measurement code only. They cannot establish a learned mask
effect. Inference-only model runs can validate reward resolution but still
cannot estimate training efficacy. Only independently seeded training runs can
support the factorial estimand.

## Evidence roots

- prior terminal evidence: `../structured_action_supervision_v2/stages/B3/`
- novelty evidence: `stages/P0/`
- frozen task instrument: `stages/P1/`
- terminal inference result: `stages/P2/stage_result.json`
- terminal report and hook: `stages/P2/REPORT.md`,
  `stages/P2/hook_result.json`
- raw evidence manifest: `stages/P2/raw/MANIFEST.json`
- terminal successor handoff: `successors/rist_v1_1/START_HERE.md`
