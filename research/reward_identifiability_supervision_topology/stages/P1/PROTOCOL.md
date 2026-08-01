# P1 CPU Task and Measurement Freeze

Protocol: `RIST-P1-v1.0`

Frozen: `2026-08-01`

## Purpose

Construct a deterministic research-only tool workflow that separates action
sequence diversity from strict-reward resolution before any model is queried.
This stage validates the instrument; it does not estimate a training effect.

## Factorial task space

Each task has exactly four ordered action opportunities and one exact oracle
call at each turn. Cross five binary task factors:

1. `constraint_slack`: `0` or `2` locally plausible non-oracle arguments;
2. `distractor_tools`: `0` or `2` irrelevant tools;
3. `dependency_depth`: `0` or `3` arguments hidden behind prior observations;
4. `argument_options`: `1` or `4` possible arguments at dependent turns;
5. `tool_choice_mode`: `forced` or `free`.

All 32 cells must appear in every split. Use three replicates per cell for
train and one per cell for qualification and held-out:

- train: 96 tasks;
- qualification: 32 tasks;
- held-out: 32 tasks.

Dataset seed: `20260801`. Generator and schema version:
`rist-workflow-v1`.

## Exact oracle and reward

The strict oracle is the unique four-call sequence matching every expected
tool name and argument object. Strict reward is binary: `1` only for the exact
oracle and `0` otherwise. Invalid or truncated sequences remain failures.

The CPU analytic policy is uniform over the explicitly enumerated admissible
name/argument choices. For a task with exact-oracle probability `p`, group-size
eight reward resolution is:

`M(p) = 1 - p^8 - (1-p)^8`.

This is a construction diagnostic, not a claim about a language model.

## P1 gates

Pass only if both train (development) and qualification (held-out Arbor merge
gate) satisfy every check:

1. expected task count;
2. all 32 factorial cells represented with equal replication;
3. unique content signatures within and across splits;
4. exactly four oracle actions and one unique strict oracle per task;
5. finite analytic probabilities and exact possible-sequence counts;
6. at least one low, intermediate, and high mixed-group-probability stratum;
7. examples where action diversity is high but reward resolution is low;
8. byte-identical regeneration and content-addressed manifest.

The Arbor development evaluator may score train only. The Arbor test evaluator
may score qualification once for the baseline and once for a selected
candidate. Candidate search must not score held-out tasks. Held-out may be
written and hashed but remains unopened by candidate selection.

## Outcomes

- `PASS_P1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`
- `KILL_TASK_GENERATOR_NO_IDENTIFIABILITY_RANGE`
- `INVALID_P1_REPRODUCIBILITY_OR_LEAKAGE`

On pass, the main-conference hook remains diagnostic and opens only P2
inference qualification. GPU training or serving remains separately gated.

