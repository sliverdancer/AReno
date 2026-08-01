# ARCA P5 external natural-validation freeze

Protocol ID: `ARCA-P5-EXTERNAL-v0.1`

Freeze date: `2026-08-01`

Owner decision: P5 is opened as an independent CPU evidence gate for the TMLR
audit/tooling route. It does not imply that P4 passed. P4-v0.2 remains
`INVALID_P4_INFRASTRUCTURE_REWARD_IMPORT` and supplies no dynamic result.

## Question

Do the already frozen ARCA contract rules detect naturally occurring silent
reward-contract failures in agentic-RL systems that were not used to develop
or freeze those rules?

## Sampling frame

Candidate systems must have, at freeze time:

1. a public version-controlled repository;
2. an explicit reinforcement-learning training path for agents, multi-turn
   interaction, environments, or tool use;
3. a source-locatable reward, trace, advantage, or trainable-token boundary;
4. no prior role in ARCA P1--P3 development, held-out evaluation, or
   post-freeze replication.

The five frozen candidates are the first qualifying, non-duplicate systems
identified by the registered discovery queries and then supplemented by the
official Agent Lightning repository as a deliberately architecture-diverse
fifth system. Source HEADs were recorded with `git ls-remote` before code
inspection. Frameworks are not removed after inspection because no failure is
found.

## Natural-case inclusion rule

A case counts as a natural external failure only when all conditions hold:

- it is present in the pinned upstream source without an ARCA mutation;
- the relevant production function or an exact source-level data-flow probe is
  reproducible with deterministic CPU fixtures;
- ordinary execution at the audited boundary succeeds and produces a finite
  numeric reward, advantage, mask, or success-like record;
- the result violates one of the pre-existing F1--F6 contracts;
- a positive control demonstrates the intended distinguishable behavior;
- the case is not documented by the framework as an intentional alias,
  explicit fallback, or rejected invalid configuration.

Import failures, missing dependencies, exceptions, non-finite values, comments
without reachable code, synthetic mutations, and speculative vulnerabilities
do not count as natural failures. Public disclosure or maintainer contact is a
separate owner-authorized action and is not part of P5.

## Frozen evaluator

The rule codes and thresholds in `research/silent_reward_contracts/arca.py` are
immutable for P5. Existing development cases may be used only to confirm that
the frozen evaluator still runs. P5 observations may not change a rule,
threshold, candidate set, or inclusion criterion. Any required rule change
invalidates v0.1 and must be reported rather than silently incorporated.

## Outcomes

`PASS_P5_EXTERNAL_NATURAL_TO_PAPER` requires:

- at least two newly reproduced natural cases;
- at least one qualifying case in a previously unseen framework;
- detection of every included high-severity natural case by frozen ARCA rules;
- clean-control false-positive rate at most 0.05;
- no post-freeze evaluator or inclusion-rule changes;
- deterministic, schema-valid JSON and CSV artifacts with source hashes.

If no natural case is reproduced, return
`KILL_P5_NO_EXTERNAL_NATURAL_EVIDENCE`. If one case is reproduced, or breadth
otherwise falls short, return `NARROW_P5_INSUFFICIENT_EXTERNAL_BREADTH` and
retain only the three-system TMLR claim. Infrastructure-only failures are
reported as `BLOCKED_*` or `INVALID_*` and never converted into scientific
negatives.

## Analysis

The statistical unit is `framework x natural_case`; source files and fixture
variants are repeated evidence, not independent prevalence observations.
Report counts and exact case records. No field-wide prevalence estimate is
permitted from this purposive sample. Clean controls are reported separately
with a Wilson 95% interval; they are not pooled with natural cases.

