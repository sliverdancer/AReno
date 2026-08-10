# Collapse taxonomy

This taxonomy describes observed failure modes that make supervision-topology
effects non-estimable under group-relative objectives.

## 1. All-fail collapse

Definition: every rollout in a task group receives strict reward 0.

Observed evidence:

- RIST v3.1: Qwen3 and Gemma4 were all-fail on all eight cells.
- RIST v4.0: Qwen3 was all-fail on all eight cells.
- RIST v4.0: Gemma4 was all-fail on cells `c02` through `c07`.

Why it matters:

Group-relative objectives need reward variation within a rollout group. If all
rollouts have reward 0, the group contains no success/failure contrast, so
changing the supervised action span cannot be connected to strict task success.

## 2. All-pass collapse

Definition: every rollout in a task group receives strict reward 1.

Observed evidence:

- RIST v4.0 Gemma4 `c00`: 128/128 strict successes, zero mixed groups.

Why it matters:

All-pass is as uninformative as all-fail for group-relative advantage. It
demonstrates task solvability for a model, but not reward resolution.

## 3. Single-model mixedness without cross-model transport

Definition: one family has a local mixed group, but the same structural cell is
not mixed or high-resolution in the other family.

Observed evidence:

- RIST v4.0 Gemma4 `c01`: one mixed group and 43/128 successes.
- RIST v4.0 Qwen3 `c01`: 0/128 successes.

Why it matters:

The RIST qualification gate required common structure across model families. A
single-family ambiguous cell cannot support a robust two-family supervision
topology experiment.

## 4. Copy-code collapse

Definition: the task requires exact argument copying in a form that creates a
first-turn bottleneck before the intended multi-turn structure is exercised.

Observed evidence:

- RIST v3.1 failure taxonomy: both families mostly failed at turn 1 with
  `WRONG_CODE`.
- v3.1 was retired instead of patched because the long hash-like code format was
  a confound.

Why it matters:

The intended scientific variable was action-span supervision, not arbitrary
string-copying skill. A first-turn code bottleneck can dominate the reward and
hide all downstream topology effects.

## Design lesson

The instrument should be judged by reward-resolution transport, not only by
syntactic task validity. A task pool can have unique oracles, sealed splits,
exact receipts, and zero-retry collection while still being scientifically
invalid for group-relative supervision experiments.
