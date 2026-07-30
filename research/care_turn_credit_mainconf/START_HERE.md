# START HERE — CARe: Risk-Controlled Turn Credit for Agentic RL

Protocol ID: `CARE-P3-PILOT-v0.1`
Decision date: `2026-07-30`
Target: main-conference method paper
Current authorization: completed CPU design and deterministic qualification
GPU training status: `BLOCKED_PENDING_EXPLICIT_USER_AUTHORIZATION`

## Owner goal

Develop and falsify a method for reliable turn-level credit assignment in
multi-turn language-agent reinforcement learning. The working method, **CARe**,
routes policy-gradient credit only to turns whose credit sign is sufficiently
reliable, abstains on uncertain turns, and respects a declared supervision-token
budget.

The paper is not about the existence of AReno's three static
`trainable_turns` modes. Those modes are engineering primitives and diagnostic
controls. The proposed scientific contribution is risk-controlled selective
credit routing under noisy turn-credit estimates.

## Binding scope separation

- The old trajectory-censoring coding instrument ended at
  `KILL_CURRENT_CODING_INSTRUMENT`. It must not be retried, scaled, or used as
  evidence for this route.
- The issue #199 Tic-Tac-Toe three-arm harness remains an integration smoke
  test. On its canonical trajectories, `last_assistant` and `final_answer`
  select the same span and are not independent scientific treatments.
- Existing token counts measure supervision density. With the current dense
  implementation they do **not** establish lower GPU FLOPs or wall-clock cost.
- Tool-call arguments are model-generated actions in the current examples.
  They must not be described as environment-determined tokens.

## Current gate

`P3-CPU-FREEZE-PASSED-GPU-BLOCKED`

Read in order:

1. `LITERATURE_AUDIT.md`
2. `RESEARCH_PLAN.md`
3. `HYPOTHESES_AND_GATES.md`
4. `../../examples/agentic/trainable_turns_ablation/README.md`

Then inspect the worktree and the completed P0/P1/P2/P3 CPU decisions. The
frozen P3 runner, task, router, asset manifest, artifact collector, commands,
resource cap, and GO/KILL rule exist. Do not provision a host, download model
weights, train, serve, or use a paid API without a new explicit authorization.

For an AutoDL instance, read `p3/AUTODL_HANDOFF.md` before remote access.

## Working method in one figure

```mermaid
flowchart LR
    A["On-policy multi-turn trajectories"] --> B["Base turn-credit scorer"]
    A --> C["Sparse counterfactual audits"]
    B --> D["Trajectory-block risk calibration"]
    C --> D
    D --> E{"Credit sign reliable?"}
    E -->|"yes"| F["Route signed credit"]
    E -->|"no"| G["Abstain: mask this turn"]
    F --> H["Budgeted policy update"]
    G --> H
    H --> I["Held-out task success"]
    H --> J["Sign-error risk and coverage"]
    H --> K["Environment calls, tokens, time"]
```

## Stage state

| Stage | State | Opening condition |
|---|---|---|
| P0 full-text novelty audit | `PASS_NOVELTY_CONDITIONAL` | Completed |
| P1 formalization and CPU oracle | `PASS_THEORY_ORACLE_CONDITIONAL_TO_P2` | Completed |
| P2 AReno instrument qualification | `PASS_P2_INSTRUMENT_TO_P3_DESIGN` | Completed |
| P3 bounded GPU executability pilot | `CPU_FREEZE_PASS_GPU_BLOCKED` | Clean committed source, remote preflight, and explicit user authorization |
| P4 variance/resource freeze | `UNOPENED` | P3 is valid and non-degenerate |
| P5 confirmatory multi-seed study | `UNOPENED` | Frozen manifest plus new explicit authorization |
| P6 transfer/robustness study | `UNOPENED` | P5 primary gate passes |
| P7 paper claim gate | `UNOPENED` | All admissible evidence archived |

## First executable work

The reviewed P2/P3 scope is packaged on the dedicated
`research/trainable-turns-ablation` branch. The exact source revision is the
commit containing this document and must be recorded with `git rev-parse HEAD`
after checkout. GPU execution still requires a separate authorization after
the remote read-only preflight verifies that commit and the live provider
quote.

## Evidence and artifact roots

- Engineering harness:
  `examples/agentic/trainable_turns_ablation/`
- Per-step metric path:
  `areno/api/metrics.py`
- Proposed research artifacts:
  `research/care_turn_credit_mainconf/`

## Stop rules

- Stop at the first terminal `KILL`, `INVALID`, or `BLOCKED` decision.
- Do not repair a consumed confirmatory run or selectively rerun failed seeds.
- Do not use a pilot reward curve as efficacy evidence.
- Do not rent or start a GPU until the user approves the exact frozen command,
  model, expected GPU-hours, and spending ceiling.
