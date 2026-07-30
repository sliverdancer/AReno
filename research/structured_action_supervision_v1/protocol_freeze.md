# SAS-P0-v1.0 — Formal Research Plan

Status: `DESIGN_FROZEN_FOR_Q0_ONLY`  
Formal outcome experiment: `UNOPENED`

## 1. Research question

In deterministic multi-turn tool use where outcome reward is caused by structured
tool-call arguments, how do temporal credit breadth and supervised action content
affect held-out task success, optimization stability, and sample efficiency?

## 2. Primary estimand

For an identical initial checkpoint, task schedule, optimizer-step budget, and
seed set:

`Delta_primary = held_out_strict_success(AF) - held_out_strict_success(LF)`

The optimizer-step-matched contrast is primary. A trainable-token-matched
contrast is mandatory sensitivity analysis because masking changes effective
sequence length and GSPO normalization.

The primary hypothesis is two-sided. No directional benefit is assumed.

## 3. Factorial arms

| ID | Mode | Mask arguments | Scientific role |
|---|---|---:|---|
| `AF` | `all_assistant` | false | All action turns, full calls; primary |
| `LF` | `last_assistant` | false | Final action turn, full call; primary |
| `AN` | `all_assistant` | true | All forced names only; mechanism control |
| `LN` | `last_assistant` | true | Final forced name only; mechanism control |
| `Z0` | `final_answer` on an all-tool-call trajectory | n/a | Empty-mask/no-update integrity control |

`Z0` is not a training arm. It is a deterministic implementation check.

## 4. Environment contract

Use a research-specific derivative of the shopping task with exactly four
ordered actions:

1. retrieve candidates;
2. inspect candidate attributes;
3. validate a proposed bundle;
4. submit the bundle.

The tool name may be forced to isolate argument learning, but all arguments must
come from the raw model response. The runner must:

- preserve raw completions;
- accept exactly one expected tool call per turn;
- reject missing, multiple, malformed, unexpected, or non-executable calls;
- never synthesize calls or arguments;
- retain exact call/result pairing;
- score invalid trajectories as failures and report them separately;
- stop deterministically at the first contract violation.

## 5. Dataset contract

Create tasks from a declared combinatorial space of catalog, budget, category,
compatibility, and exclusion constraints.

- Train, qualification, and held-out partitions are split by constraint
  combination, not row ID.
- No normalized constraint signature may occur in more than one partition.
- The held-out manifest is hashed before Q1 and may not be inspected for reward
  outcomes until Q3.
- Duplicate prompts and semantically identical constraint signatures are fatal
  validation errors.
- Dataset generation seed and generator version are recorded.

## 6. Outcomes

### Primary

- strict held-out task success at the frozen optimizer-step budget.

### Secondary

- area under the strict-success curve over environment interactions;
- valid and executable call rate;
- per-turn contract compliance;
- final reward;
- cumulative trainable tokens;
- effective masked response length;
- policy loss, gradient norm, sequence ratio, and clipping diagnostics;
- wall-clock time and peak GPU memory.

Secondary outcomes do not rescue a failed primary outcome.

## 7. Reproducibility contract

Each run records:

- source commit and dirty-worktree diff hash;
- model/checkpoint identity and content hash where available;
- dataset and split manifest hashes;
- initialization, sampling, data-order, and environment seeds;
- complete resolved config;
- package versions, CUDA/runtime version, and GPU model;
- exact command and exit status;
- per-step metrics and final artifact hashes.

Until training seeds are explicit and testable, Q1 and Q3 remain blocked.

## 8. Stage gates

Every stage closure must run `stage_completion_hook.py`. The hook records
`GO_MAIN_TRACK`, `STAY_DIAGNOSTIC`, or `KILL_MAIN_TRACK`. Before Q3, a passing
stage can only remain diagnostic; pilot evidence cannot trigger a main-track
claim.

### Q0 — Design and instrument qualification

Authorized now; CPU/deterministic work only.

Required pass evidence:

- exact span/mask fixtures for all five IDs;
- distinct `AF`, `LF`, `AN`, and `LN` masks on the same canonical trajectory;
- `Z0` has zero effective trainable tokens and produces zero parameter delta;
- invalid calls are rejected with stable reason codes;
- split-integrity and duplicate-signature tests pass;
- metric schema covers every declared outcome;
- seed provenance is explicit end to end.

Failure outcome: `KILL_CURRENT_INSTRUMENT` or
`BLOCKED_PUBLIC_CONFIG_DECISION`.

### Q1 — Bounded integration pilot

Unopened. Requires Q0 pass and explicit GPU/model authorization.

- Run only `AF` and `LF`.
- Use qualification tasks only; do not consume held-out outcomes.
- Use a small, predeclared step ceiling.
- Measure base executability, reward variance, throughput, memory, and seed
  variance.
- The pilot is engineering/variance evidence, never an efficacy result.

Minimum qualification:

- at least 95% parser-valid and executable calls before training;
- non-degenerate task reward across qualification prompts;
- complete metrics for every attempted trajectory;
- no fabricated or silently repaired trajectories;
- no out-of-memory or non-finite optimization event.

Failure outcome: `KILL_CURRENT_INSTRUMENT`.

### Q2 — Power and resource freeze

Unopened. Estimate variance only from Q1 qualification data.

- Freeze a minimum detectable absolute success-rate difference of 10 percentage
  points unless the user approves a different practical threshold.
- Target at least 80% power with two-sided alpha 0.05 for the primary contrast.
- Require at least three independent training seeds per primary arm; use more if
  the power calculation and resource ceiling permit.
- Freeze the final seed list, step budget, task count, exclusions, and analysis
  script before Q3.
- If adequate power does not fit the approved resource ceiling, return
  `KILL_UNDERPOWERED_CONFIRMATORY_ROUTE`.

### Q3 — Confirmatory experiment

Unopened. Requires a new immutable manifest and explicit GPU authorization.

- Execute the frozen arms and seed schedule without selective reruns.
- Analyze all attempted seeds under the frozen invalid-run policy.
- Report seed-level estimates and paired task-level uncertainty.
- Correct the two mechanism-control contrasts as a declared family; the primary
  `AF` versus `LF` contrast remains singular.
- A failed or null result is final evidence, not a trigger for post-hoc repair.

### Q4 — Claim gate

Unopened.

Possible outcomes:

- `PASS_DIAGNOSTIC_EFFECT`: evidence supports a bounded claim about the tested
  task, model, algorithm, and budgets.
- `PASS_NULL_OR_NEGATIVE`: report a reproducible null/negative result with the
  same scope.
- `INCONCLUSIVE`: uncertainty crosses both zero and the practical-effect
  threshold.
- `INVALID`: protocol, provenance, or instrument failure; no scientific claim.

## 9. Analysis plan

- Report absolute success proportions and arm differences with 95% confidence
  intervals.
- Treat training seed as the independent replication unit.
- Use paired task evaluation within each seed and hierarchical/bootstrap
  intervals that preserve the seed structure.
- Plot every seed; do not show only aggregate best runs.
- Report invalid trajectories in the denominator for strict success.
- Provide both optimizer-step-matched and trainable-token-matched curves.
- Do not infer generality beyond the tested model, task family, algorithm, and
  budget.

## 10. Claim boundary

Permitted before Q3:

> AReno exposes static assistant-span and tool-argument masking mechanisms, and
> we have frozen a protocol to test their causal effects.

Not permitted before Q3:

- the method improves performance or convergence;
- argument masking improves credit assignment;
- the approach is novel or first;
- results generalize to open-ended agents;
- smoke-test success is scientific evidence.

## 11. Change control

Any change to the primary estimand, held-out split, arm definition, exclusion
rule, or analysis after Q3 begins invalidates the confirmatory label. Create a
new protocol version and leave the consumed result archived.
