# Reward Identifiability x Supervision Topology

Branch: `research/reward-identifiability-supervision-topology`

Protocol family: `RIST-v1`

Status: `P0_PROTOCOL_FROZEN`

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
- current protocol: `stages/P0/PROTOCOL.md`
- current search evidence: `stages/P0/`
- future generated tasks: `stages/P1/`

