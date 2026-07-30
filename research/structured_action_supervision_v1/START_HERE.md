# START HERE — Structured Action-Span Supervision (SAS)

Protocol ID: `SAS-P0-v1.0`  
Branch: `research/trainable-turns-ablation`  
Base commit: `d5a4f5d53e2f0241b945f980126f88968b4b5af4`  
Decision date: `2026-07-29`

## Owner goal

Determine whether selecting which assistant action spans receive policy-gradient
supervision changes learning in deterministic multi-turn tool use, while separating:

1. temporal credit breadth — all action turns versus the final action turn; and
2. action content — full tool call versus tool-name-only supervision.

The study is a falsification-first replication/ablation. It is **not** currently
authorized to claim a new static masking method, better convergence, or superior
agent performance.

## Binding decision

- `PASS_FEASIBILITY_CONDITIONAL_ACTION_ALIGNED_REDIRECT`
- `KILL_ORIGINAL_THREE_ARM_NOVELTY_CLAIM`
- `BLOCK_CONFIRMATORY_TRAINING_UNTIL_Q0_Q1_PASS_AND_EXPLICIT_GPU_AUTHORIZATION`

The existing three-arm Tic-Tac-Toe harness remains useful only as an engineering
smoke test. It must not be reported as scientific evidence because
`last_assistant` and `final_answer` target the same second response, while the
rewarded move is emitted in the first response.

## First-turn behavior for the next agent

1. Read, in order:
   - `MAIN_CONFERENCE_RESEARCH_PLAN.md`
   - `main_conference_literature_addendum.md`
   - `gate_decision.md`
   - `feasibility_audit.md`
   - `literature_search_log.md`
   - `protocol_freeze.md`
   - `hypothesis_matrix.csv`
   - `MAIN_TRACK_HOOK.md`
2. Inspect the current worktree before editing. Preserve all unrelated local
   changes and do not modify the separate root-level `START_HERE_RESEARCH.md`.
3. Execute only the next open gate, `Q1-BOUNDED-INTEGRATION-PILOT`, using the
   frozen `stages/Q1/pilot_manifest.json`.
4. Do not run training, serving, model downloads, or GPU work without explicit
   user authorization. The current local session exposes no CUDA device.
5. Public seed configuration and CLI changes were explicitly authorized and are
   implemented. Any further public-config, CLI, or dependency change still
   requires prior authorization under `AGENTS.md`.
6. Close every completed stage through `stage_completion_hook.py`; do not open
   the next stage if its main-track assessment is missing.

## Evidence roots

- Mask implementation: `areno/api/agentic.py`
- Policy-only advantage and masked loss path:
  `areno/api/trainers/policy_only.py`
- GSPO loss geometry: `areno/api/loss_fns/gspo.py`
- Response-layout masking: `areno/api/loss_fns/layout.py`
- Existing engineering harness:
  `examples/agentic/trainable_turns_ablation/`
- Candidate multi-turn environment: `examples/agentic/shopping/`

## Current state

| Stage | State | Meaning |
|---|---|---|
| P0 literature and local feasibility audit | `PASS_WITH_REDIRECT` | The narrow causal question is testable; the original novelty framing is not. |
| Q0 design and instrument qualification | `PASS` | Instrument, strict reward, split isolation, zero-signal guard, and seed chain passed CPU qualification; hook decision is `STAY_DIAGNOSTIC`. |
| Q1 bounded integration pilot | `PREPARED_GPU_UNAUTHORIZED` | AF/LF × seeds 1101/2202 is frozen on qualification data; no model or GPU run has started. |
| Q2 power/resource freeze | `UNOPENED` | Uses pilot-only variance; no held-out outcomes may be consumed. |
| Q3 confirmatory experiment | `UNOPENED` | Requires a new frozen manifest and explicit GPU authorization. |
| Q4 paper/public claim | `UNOPENED` | Permitted only from admissible Q3 evidence. |

The main-conference route is governed by `SAS-MAIN-v1.0`. It does not replace
the frozen Q1 manifest or promote Q0 evidence. The generic adaptive-span method
claim remains unopened pending the post-Q1 N1 novelty gate.

## Stop rule

Stop at the first `KILL` or `BLOCKED` gate. Do not repair a consumed
confirmatory result, selectively rerun seeds, or reinterpret an engineering
smoke test as scientific evidence.
