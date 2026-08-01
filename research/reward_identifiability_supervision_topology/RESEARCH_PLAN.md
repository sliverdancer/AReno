# RIST-v1 Frozen Research Plan

Date: `2026-08-01`

Status: `P2_TERMINAL_INVALID_INFRASTRUCTURE`

Terminal note: P0 and P1 passed their frozen gates. P2 stopped under
`REPEATED_SERVER_FAILURE` before any scientific response and returned
`INVALID_P2_INFRASTRUCTURE`. P3-P5 remain unopened. A new attempt requires a
newly frozen protocol and fresh authorization; it cannot be spliced into
`RIST-P2-v1.0`.

## Research question

In multi-turn tool-agent learning, when are contrasts between supervision over
all versus only the final action and full-call versus name-only content
identifiable under strict outcome rewards?

## Primary estimand

Let `S` denote supervision topology (`AF`, `LF`, `AN`, `LN`) and `R` a
predeclared reward-resolution stratum measured before training. The primary
estimand is the interaction between `S` and `R` on strict task success, with
stability and sample efficiency as prespecified secondary outcomes.

The study is two-sided. It does not require all-turn or full-call supervision
to win.

## Competing hypotheses

### H1 — Reward-resolution identifiability

Mask effects are observable only when tasks generate sufficient within-task
reward variation. Prediction: low-resolution strata have high advantage
collapse and near-zero estimable mask contrasts even when action sequences
vary; intermediate strata produce non-zero advantages and lower uncertainty.

Falsifier: reward-resolution measures fail to predict gradient-bearing groups
or precision after controlling for task and model.

### H2 — Temporal coverage

When early actions causally constrain later success, all-action supervision
outperforms last-action supervision. Prediction: the AF-LF and AN-LN contrasts
increase with dependency depth.

Falsifier: the interaction is practically negligible with intervals excluding
the prespecified minimum effect across qualified models and environments.

### H3 — Argument information

Full-call supervision matters when correct arguments carry task-specific
information not recoverable from the tool name. Prediction: AF-AN and LF-LN
contrasts increase with argument entropy.

Falsifier: full versus name-only contrasts remain practically negligible in
high-argument-entropy strata with adequate power.

### H4 — Objective-geometry alternative

Any apparent all-action advantage is explained by greater trainable-token mass
rather than causal coverage. Prediction: the step-matched effect shrinks or
reverses under cumulative-trainable-token matching.

Falsifier: the temporal interaction survives both step-matched and
token-matched analyses.

## Stage plan and gates

### P0 — Direct-substitute novelty review

Search arXiv, ACL Anthology, OpenAlex, Crossref, OpenReview, and citation chains
for work on reward diversity/collapse, agentic credit assignment, task
difficulty, tool-call structure, and supervision masking.

- `KILL_DIRECT_SUBSTITUTE`: a work already makes substantially the same causal
  interaction or benchmark contribution.
- `PASS_NOVELTY_CONDITIONAL`: adjacent components are occupied, but the joint
  interaction study remains open.
- `BLOCKED_FULL_TEXT_OR_ARTIFACT`: a likely substitute cannot be inspected.

### P1 — CPU task and measurement freeze

Build a deterministic research-only task generator with fresh hashed
train/qualification/held-out splits. Vary constraint slack, distractor count,
action dependency depth, argument entropy, and forced/free tool choice without
using model outcomes to select tasks.

Pass only if:

- generation is byte-reproducible;
- split signatures are disjoint;
- every task is executable and has a unique strict oracle;
- strata cover low/intermediate/high analytic reward resolution;
- action diversity and reward diversity are measured separately;
- qualification and held-out remain unopened by search.

### P2 — Inference-only qualification

Use at least two checkpoints from at least two model families. Qualification
tasks only. Estimate conditional reward entropy, mixed-group probability,
advantage-collapse rate, tool executability, and action-sequence diversity.

Pass only if at least two predeclared strata are separable in reward resolution
and each intended training stratum has adequate mixed groups. This stage needs
separate GPU/serving authorization.

### P3 — Factorial training pilot

Run `AF/LF/AN/LN` on qualification-only tasks with three paired seeds. This is
an instrumentation and variance pilot, not a confirmatory result.

### P4 — Prospective power and resource freeze

Use only P3 variance estimates to select the independent training-seed count.
Freeze minimum detectable effect, exclusions, model/environment matrix,
compute ceiling, and statistical analysis before opening confirmatory data.

### P5 — Confirmatory study

Complete the powered factorial across at least three environments, three
checkpoints spanning two model families, and two policy-optimization
algorithms. Report strict success, catastrophic-run rate, paired uncertainty,
sample-efficiency curves, and token-matched sensitivity.

## Main-conference hook

Run after every completed stage.

- P0 or P1 failure: `KILL_CURRENT_ROUTE`.
- P1 pass: `STAY_DIAGNOSTIC_OPEN_INFERENCE_QUALIFICATION`.
- P2 pass: `STAY_DIAGNOSTIC_OPEN_FACTORIAL_PILOT`.
- P3 pass with feasible power: `UPGRADE_MAIN_TRACK_CANDIDATE`.
- P5 may return `GO_MAIN_TRACK` only if the primary interaction is practically
  meaningful, its 95% interval excludes zero, cross-setting evidence is
  consistent, direct baselines are complete, and artifacts reproduce.

## Bias controls

- The terminal B3 data motivate hypotheses but never tune RIST tasks.
- Split assignment is salted and content-addressed before model execution.
- Qualification may guide feasibility; held-out admits the final claim once.
- Invalid trajectories stay in strict-success denominators.
- Tasks, rollout samples, and checkpoints are not counted as independent
  training-seed replications.
- Negative or null outcomes are terminal under the same reporting standard as
  positive outcomes.
