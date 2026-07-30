# Feasibility Audit

Audit date: `2026-07-29`  
Scope: local implementation, existing agentic examples, and direct research
neighbors  
Outcome: `PASS_WITH_REDIRECT`

## Answer first

The scientific question is feasible only after reframing it from “a novel
trainable-turn feature” to “a controlled causal ablation of action-span
supervision.” The code can express the intervention, but the current task and
runner cannot support a defensible outcome claim.

## What is already sufficient

### Intervention mechanism

`areno/api/agentic.py` defines `all_assistant`, `last_assistant`, and
`final_answer`, together with optional tool-call argument masking. The policy
trainer propagates the resulting loss mask into token-level optimization.

This supports a 2×2 diagnostic:

| Arm | Temporal breadth | Content retained |
|---|---|---|
| `AF` | all assistant action turns | full tool calls |
| `LF` | last assistant action turn | full tool call |
| `AN` | all assistant action turns | forced tool names only |
| `LN` | last assistant action turn | forced tool name only |

`AN` and `LN` are mechanism/negative controls when tool names are externally
forced. They are not candidate improvements.

### Candidate environment

The shopping example has four ordered tool interactions and a final outcome
reward. Its structure can isolate early versus late action arguments better than
the single-action Tic-Tac-Toe and DuelGrid examples.

## Why the existing experiment is inadmissible

### The Tic-Tac-Toe arms are not three distinct treatments

The ablation runner adds a second plain-text response after the rewarded
`choose_square` call. Its documentation intentionally makes
`last_assistant` and `final_answer` select that same second span. Therefore:

- only two distinct masks are compared;
- the `final_answer`/`last_assistant` arm trains post-action explanation tokens;
- the reward-causing move in the first response is excluded;
- any difference confounds supervision density with causal action alignment.

The 10-step run is retained only as an integration smoke test.

### The shopping runner fabricates missing calls

The current shopping runner inserts a placeholder tool call with empty arguments
when the model emits no call. That behavior is acceptable neither for scientific
measurement nor for trajectory provenance. A research runner must record the
trajectory as invalid and preserve the raw model response.

### The shopping dataset is not a held-out distribution

The generator repeats four templates with different IDs. An ID split would
therefore leak the same constraint structures across partitions. The research
instrument needs a combinatorial task generator and a split keyed by unseen
constraint combinations, not row IDs.

### Reproducibility is incomplete

The training CLI/config does not expose an explicit training seed. Sampling has
seed support elsewhere, but confirmatory training cannot open until initialization,
sampling, data order, and environment seeds are all plumbed and recorded.
Changing public config or CLI surfaces requires separate user approval.

### The intervention changes objective geometry

GSPO normalizes sequence-level quantities using the masked response length.
Sparse masks therefore alter both which tokens receive gradient and the
effective sequence normalization. Results cannot be described as a pure
“same-objective, fewer-token” comparison.

The protocol must report:

- optimizer steps and environment interactions;
- trainable tokens per trajectory and cumulatively;
- effective masked response length;
- gradient and ratio diagnostics;
- both step-matched and trainable-token-matched curves, with the primary
  estimand declared in advance.

## Threats to validity and controls

| Threat | Consequence | Required control |
|---|---|---|
| Reward is broadcast across selected tokens | Credit is not truly localized | Describe the study as span selection, not token-level causal attribution |
| Forced tool names | Name-only arms have nearly no decision content | Treat them as negative controls |
| Invalid-call synthesis | Inflated executability and broken provenance | Reject and count invalid calls |
| Four repeated task templates | Test leakage | Split on unseen constraint combinations |
| Different masked lengths | Compute/objective confound | Report both budgets and effective lengths |
| No explicit training seed | Irreproducible seed effects | Block Q1 until seed provenance exists |
| Small pilot | High variance and winner’s curse | Use pilot only for qualification and power planning |
| Direct prior art | Novelty overclaim | Position as replication/diagnostic unless later evidence supports a narrower gap |

## Feasibility judgment

### Technical

`PASS_CONDITIONAL`: the current mask machinery is adequate, but a
research-specific runner, dataset split, seed provenance, and integrity tests
are required.

### Scientific

`PASS_FOR_DIAGNOSTIC_CAUSAL_ABLATION`: a narrow question about temporal breadth
and action content is identifiable under a deterministic tool protocol.

### Novelty

`FAIL_BROAD_NOVELTY`: direct and adjacent prior work occupies action-focused,
turn-level, transition-level, and intra-call supervision territory.

### Resource

`BLOCKED_PENDING_PILOT`: no formal run count or GPU budget is authorized until
Q1 measures invalid-rate, variance, throughput, and memory. If an adequately
powered study cannot fit the approved ceiling, stop with
`KILL_UNDERPOWERED_CONFIRMATORY_ROUTE`.

