# Structured Action-Span Supervision — Main-Conference Research Plan

Plan ID: `SAS-MAIN-v1.0`  
Parent protocol: `SAS-P0-v1.0`  
Status: `PLAN_FROZEN_BEFORE_Q1_EXECUTION`  
Date: `2026-07-30`  
Current executable stage: `Q1-BOUNDED-INTEGRATION-PILOT`

## 1. Answer first

The project is a **conditional main-conference candidate**, not yet a
main-conference result.

The defensible primary route is an empirical and benchmark contribution:

> Structured action-span supervision is an undercontrolled factor in
> multi-turn tool-agent policy optimization. A strict factorial protocol can
> identify when temporal credit breadth and intra-call content change exact
> task success, stability, and sample efficiency.

Static masking and generic action-token reweighting are not claimed as novel.
An adaptive method is an optional later route and may open only after Q1
produces a real failure mode, a new direct-substitute search passes, and the
method makes a falsifiable prediction beyond existing action weighting,
turn-level credit, process reward, or observation supervision.

## 2. Target contribution ladder

### C1 — Required: reproducible causal diagnostic

- A strict multi-turn tool runner that never fabricates or repairs calls.
- A 2×2 intervention separating temporal breadth from intra-call content.
- Step-matched and trainable-token-matched estimates.
- Seed-level uncertainty, invalid-call accounting, and frozen held-out splits.
- Negative and null results remain publishable artifacts.

### C2 — Required for main-track: generalization map

- At least three task environments with different horizons and action
  structures.
- At least three compatible checkpoints spanning at least two model families.
- At least two policy-optimization algorithms.
- A prespecified analysis of horizon, argument complexity, and supervision
  density as moderators.

### C3 — One of the following is additionally required

1. **Benchmark route:** a reusable supervision-topology evaluation suite with
   strict validators, manifests, and reporting tools; or
2. **Method route:** a genuinely new intervention motivated by the causal
   results and superior to direct contemporary baselines under equal budgets.

C1 alone is a diagnostic paper. C1+C2 without C3 is likely workshop/findings
scope. C1+C2+C3 is the intended main-conference package.

## 3. Research questions

### RQ1 — Temporal breadth

Holding checkpoint, task order, optimizer steps, rollout budget, and seed
constant, does supervising all assistant action turns rather than only the last
assistant action turn change strict task success?

### RQ2 — Intra-call content

Does retaining model-generated tool arguments in supervised spans add useful
learning signal beyond retaining tool names alone?

### RQ3 — Mechanism and boundary conditions

Are observed differences explained by causal action coverage, noisy-gradient
dilution, objective geometry caused by masked length, or protocol/exploration
effects?

### RQ4 — External validity

Which effects replicate across task horizon, argument complexity, environments,
models, and algorithms?

### RQ5 — Method opportunity

Only if a stable and predictable failure mode exists: can a new, auditable
supervision policy improve strict success or sample efficiency over static
masks and direct literature baselines?

## 4. Competing mechanistic hypotheses

The complete machine-readable hypothesis set is in
`main_conference_hypotheses.csv`.

### M1 — Causal-coverage hypothesis

Early tool arguments causally determine later state. Supervising every action
turn should improve learning as causal distance and horizon increase.

Prediction: the `AF − LF` contrast becomes more positive with longer
action-to-outcome distance, while the difference contracts on one-step tasks.

Falsifier: no practically meaningful temporal-breadth effect under long-horizon,
argument-sensitive tasks with adequate precision.

### M2 — Dilution/interference hypothesis

Broadcast outcome reward gives noisy or conflicting gradients to early actions.
Restricting supervision to later actions can improve stability or success.

Prediction: `LF` has lower seed variance or better strict success when early
actions are weakly identifiable or highly stochastic.

Falsifier: `LF` never improves stability or success in the prespecified
high-ambiguity strata.

### M3 — Objective-geometry hypothesis

Apparent semantic effects arise mainly because masking changes effective
response length and GSPO normalization.

Prediction: step-matched effects materially shrink or reverse under
trainable-token matching and masked-length adjustment.

Falsifier: the direction and practical magnitude remain stable under both
budgets and across an algorithm with different normalization.

### M4 — Protocol/exploration hypothesis

Differences arise from JSON validity, tool-call formatting, or exploration
rather than credit allocation.

Prediction: treatment effects are mediated by invalid-call rates and disappear
among executable trajectories.

Falsifier: strict-success effects remain after including every invalid attempt
in the denominator and are accompanied by task-semantic rather than format-only
improvements.

## 5. Treatments and controls

| ID | Temporal span | Supervised content | Role |
|---|---|---|---|
| `AF` | all assistant action turns | full tool calls | primary |
| `LF` | last assistant action turn | full tool call | primary |
| `AN` | all assistant action turns | forced names only | mechanism control |
| `LN` | last assistant action turn | forced name only | mechanism control |
| `Z0` | empty effective span | none | no-update integrity control |

`Z0` is never a performance arm. `AN` and `LN` are controls in environments
where tool names are forced; they are not candidate improvements.

Direct literature baselines are selected only after implementation and license
feasibility checks. The minimum baseline families to audit are:

- uniform/all-token policy optimization;
- action-token reweighting;
- turn-level or transition-level credit assignment;
- process/dense reward supervision where a faithful implementation exists.

Unavailable public code is `BLOCKED_BASELINE`, not a failed baseline.

## 6. Environment ladder

### E0 — SAS Shopping, deterministic structured actions

Already qualified in Q0. Four ordered tool calls, exact state reward,
constraint-signature split, and strict invalid-call handling.

### E1 — Stateful planning/game environment

Candidate source: an AReno-local stateful example such as DuelGrid. It must be
converted to the same no-fabrication contract and must expose at least three
causally distinct action turns. Reusing the existing runner without scientific
qualification is forbidden.

### E2 — Realistic interactive tool environment

Selected after Q1 using these frozen criteria:

- public and legally usable task/evaluator artifacts;
- deterministic or auditable state transitions;
- exact executable success rather than an LLM-only judge;
- support for multiple causally relevant tool calls;
- feasible ModelScope/local asset path;
- no held-out leakage during adaptation.

If no E2 meets all criteria, main-track generalization is
`BLOCKED_ENVIRONMENT`, not satisfied by relabeling a synthetic task.

### Moderator grid

Within environments, tasks must vary prespecified moderators:

- action horizon: short / medium / long;
- action-to-reward distance;
- argument arity and value entropy;
- dependency depth between calls;
- recoverability after an early error.

Moderator bins are defined from task generators before outcome inspection.

## 7. Model and algorithm ladder

Q1 remains fixed to `Qwen/Qwen3-0.6B`, GSPO, and two seeds. It is not part of
the confirmatory main-track evidence unless explicitly included by the Q2
manifest.

Q2 selects the confirmatory matrix using only verified AReno adapters and
ModelScope assets:

- at least three checkpoints;
- at least two model families;
- at least one checkpoint small enough for full replication;
- no model selected because it produced a favorable pilot result.

Algorithms:

- GSPO is the anchor algorithm;
- GRPO is the required cross-algorithm replication;
- PPO is optional only if Q1/Q2 resource and stability evidence justify it.

No adapter, checkpoint, or algorithm is declared supported until its definition
and CPU/GPU qualification have been read and run.

## 8. Outcomes and estimands

### Primary outcome

Strict task success: every required call is present, executable, ordered, and
leads to the exact final task state. Invalid attempts remain failures.

### Primary anchor estimand

At the Q2-frozen anchor environment/model/algorithm and optimizer-step budget:

`Delta_anchor = P(strict success | AF) - P(strict success | LF)`

The test is two-sided. The current minimum practically meaningful absolute
difference is 10 percentage points. Q2 may increase this threshold but may not
reduce it after inspecting held-out outcomes.

### Mandatory sensitivity estimand

The same contrast at matched cumulative trainable tokens. A material sign
reversal blocks a simple semantic-credit claim and supports M3.

### Secondary outcomes

- area under the strict-success curve over environment interactions;
- interactions required to reach a frozen success threshold;
- valid/executable call rate and stable invalid reason counts;
- per-turn semantic argument accuracy;
- seed-level variance and catastrophic-run rate;
- trainable tokens, masked length, ratio/clipping and gradient diagnostics;
- throughput, wall time, and peak GPU memory.

Secondary outcomes cannot rescue a failed primary anchor claim.

## 9. Statistical analysis plan

### Replication unit

The independent training seed is the replication unit. Tasks are repeated
measurements nested within a seed and may not be treated as independent training
runs.

### Primary inference

- Pair AF/LF runs by initialization/data/sampling seed where technically valid.
- Report seed-level effects and every seed trajectory.
- Estimate absolute success differences with a hierarchical bootstrap that
  resamples seeds first and tasks second.
- Report 95% confidence intervals and exact attempted-run counts.
- Do not replace failed seeds; apply the frozen invalid-run policy.

### Generalization analysis

Use a prespecified hierarchical binary-outcome model or stratified bootstrap
with treatment, environment, model family, algorithm, horizon, argument
complexity, and their declared interactions. The final method is selected and
simulation-tested in Q2 before held-out execution.

### Multiple comparisons

- The AF/LF anchor contrast is the sole primary test.
- AF/AN and LF/LN form one mechanism-control family and use Holm correction.
- Moderator and per-environment analyses are secondary unless promoted in a new
  protocol before Q3.
- Exploratory analyses are labeled and cannot trigger the main-track hook.

### Power

Q1 estimates feasibility, runtime, invalid rates, and rough variance only.
Q2 conducts simulation-based prospective power using qualification data:

- target power: at least 0.80, preferably 0.90 if affordable;
- two-sided alpha: 0.05;
- minimum: five independent seeds per primary confirmatory cell for main-track
  execution, unless simulation requires more;
- no post-hoc power calculation;
- if the approved resource ceiling cannot meet power,
  `KILL_UNDERPOWERED_MAIN_TRACK`.

## 10. Staged execution and gates

### Q0 — Instrument qualification

Status: `PASS_Q0_INSTRUMENT`  
Hook: `STAY_DIAGNOSTIC`

### Q1 — Bounded GPU integration pilot

Status: `PREPARED_GPU_UNAUTHORIZED`

Execute exactly the frozen AF/LF × seeds 1101/2202 manifest on qualification
data. Pass requires:

- all four commands complete without OOM or non-finite optimization;
- no fabricated/repaired calls;
- complete metrics and raw provenance;
- parser/executable validity sufficient to estimate task behavior;
- non-degenerate rewards and measurable resource usage;
- arm masks remain distinct in emitted metrics.

Q1 does not test efficacy and cannot return `GO_MAIN_TRACK`.

### N1 — Post-pilot novelty re-audit

Open only after Q1 passes. Search direct substitutes for the observed mechanism.

- If results only reproduce existing action weighting or turn-level credit:
  keep the benchmark/diagnostic route and kill a method claim.
- If a proposed method is directly substituted: `KILL_METHOD_ROUTE`.
- If no direct substitute is found and the method has a discriminating
  prediction: `PASS_METHOD_NOVELTY_CONDITIONAL`.

### Q2 — Power, matrix, and resource freeze

Freeze:

- environments and their qualification results;
- model/algorithm cells;
- confirmatory seed list;
- optimizer-step and trainable-token budgets;
- exclusions and invalid-run handling;
- statistical code and synthetic-null checks;
- maximum GPU-hours and storage ceiling;
- exact Q3 command manifest and hashes.

No held-out model outcomes may be inspected.

### Q3-A — Primary anchor confirmation

Run the singular frozen AF/LF anchor. Stop on protocol invalidity. A null,
negative, or inconclusive result is retained and reported.

### Q3-B — Generalization matrix

Open only if Q3-A is valid, regardless of sign, and the Q2 budget remains
admissible. Complete the frozen environment/model/algorithm cells without
selective reruns.

### Q3-C — Mechanism controls

Run AN/LN and moderator analyses only in the prespecified subset. Establish
whether causal coverage, dilution, geometry, or protocol effects best explain
the result.

### M1 — Optional method stage

This stage is unopened. It requires:

- a valid Q3 failure mode or stable moderator;
- `PASS_METHOD_NOVELTY_CONDITIONAL`;
- a frozen method hypothesis and direct baselines;
- a new protocol and fresh, unconsumed held-out split.

The current Q3 held-out set may not be reused to validate a method designed from
Q3 outcomes.

### Q4 — Main-track hook and paper gate

`GO_MAIN_TRACK` requires every criterion in `MAIN_TRACK_HOOK.md`, including
valid Q3 evidence, practical effect, uncertainty excluding zero, multi-
environment/model/algorithm completion, direct baselines, sufficient seeds,
reproducible artifacts, and a benchmark or method contribution beyond static
masking.

Otherwise the decision is `STAY_DIAGNOSTIC` or `KILL_MAIN_TRACK`.

## 11. Main-track minimum matrix

The final count is determined by Q2 power simulation. The planning floor is:

| Block | Environments | Models | Algorithms | Arms | Seeds |
|---|---:|---:|---:|---:|---:|
| Anchor | 1 | 1 | 1 | AF/LF | ≥5 |
| Environment replication | 3 | 1 | 1 | AF/LF | ≥5 |
| Algorithm replication | 3 | 1 | 2 | AF/LF | ≥3 |
| Model replication | 3 | ≥3 across ≥2 families | 1 | AF/LF | ≥3 |
| Mechanism subset | ≥2 | ≥1 | ≥1 | AF/LF/AN/LN | ≥3 |

This is a planning floor, not authorization to run the full Cartesian product.
Q2 removes duplicated cells and freezes an efficient blocked design.

## 12. Bias and validity controls

| Threat | Control | Failure consequence |
|---|---|---|
| Winner's curse from Q1 | qualification-only pilot; no efficacy promotion | Q1 cannot support paper claim |
| Held-out leakage | hashed manifests; no outcome inspection before Q3 | `INVALID` |
| Seed cherry-picking | frozen seed list; all attempted seeds reported | `INVALID` |
| Masked-length confound | dual budgets and length diagnostics | block semantic-credit claim |
| Invalid-call repair | strict runner and raw response retention | `KILL_INSTRUMENT` |
| Model/algorithm selection bias | Q2 selection rules independent of favorable outcomes | `INVALID` |
| Multiple testing | single primary contrast; corrected mechanism family | downgrade to exploratory |
| Synthetic-only generalization | qualified E2 requirement | no main-track generalization |
| Direct prior art | N1 re-audit and faithful baselines | kill method novelty |
| Hardware nondeterminism | environment capture and independent seeds | no bitwise claim |

## 13. Reproducibility and artifact contract

Every run records:

- branch, commit, dirty diff hash, and exact command;
- model source, revision, local content hash where available;
- dataset/split hashes and task-generator version;
- initialization, data-order, rollout, and environment seeds;
- resolved configuration and algorithm/loss identity;
- Python, package, PyTorch, CUDA, driver, and GPU identity;
- raw generations, validated trajectories, metrics, exit status, runtime, and
  peak memory;
- run manifest and post-run artifact hashes.

Analysis scripts must pass synthetic-null, known-effect, missing-run, duplicate,
and schema-validation tests before Q3.

## 14. Resource policy

No GPU-hour estimate is treated as credible before Q1 measures actual
throughput and memory. Q2 computes:

`GPU hours = sum(cell runs × measured hours/run × safety factor)`

The safety factor and maximum budget are frozen before Q3. OOM-driven
hyperparameter changes create a new qualification attempt; they are not silently
patched into a consumed confirmatory run.

## 15. Paper claim ladder

### Before Q1

Permitted: implementation and protocol qualification only.

### After valid Q1

Permitted: bounded feasibility, resource, and variance observations on
qualification data.

### After valid Q3

Permitted claims are exactly those supported by the frozen estimands and
replication matrix. No “first”, universal agent claim, or deployable superiority
language is allowed without separate evidence.

### Candidate title

**Structured Action-Span Supervision in Multi-Turn Tool-Agent Reinforcement
Learning: A Controlled Study of Temporal Breadth and Intra-Call Content**

## 16. Immediate next action

Wait for the GPU handoff described in `GPU_HANDOFF_TEMPLATE.md`. Then:

1. inspect the host without changing it;
2. verify CUDA/PyTorch/storage and the clean branch checkout;
3. estimate whether Q1 fits;
4. request/confirm explicit authorization to execute the frozen four-run pilot;
5. run Q1 without changing its manifest;
6. write `stage_result.json` and invoke the mandatory hook.

