# RIST token-budget API CPU freeze

Protocol: `RIST-TOKEN-BUDGET-API-v1.0`

Status: `CPU_IMPLEMENTED_GPU_UNEXECUTED`

## Purpose

Provide the public stopping primitive required for a prospective independent
token-budget schedule. This stage is engineering instrumentation, not a
training result and not evidence that the supervision interaction survives
token matching.

## Frozen semantics

- Public surface: `TrainerConfig.max_trainable_tokens` and
  `areno train --max-trainable-tokens`.
- Supported algorithms: GSPO and GRPO only.
- Explicit `gradient_accumulation_steps` is rejected. Each accepted backend
  event must represent exactly one completed optimizer step.
- Effective tokens are causal next-token positions `1..length-1` that remain
  enabled after both prompt and loss masks.
- Zero-effective-token batches do not call the engine optimizer and do not
  advance the budget.
- Stop only after the first complete optimizer step at or above the target.
  Overshoot is allowed, measured, and immutable.
- A fresh run must begin at optimizer global step 1; subsequent identities
  must be contiguous.
- Existing terminal evidence or target-matched terminal checkpoints cause a
  fail-closed preflight before runtime initialization.
- On success, save the terminal checkpoint, hash its complete file manifest,
  then atomically create `token_budget_terminal.json` without overwrite.
- The evidence records target/pre/post/step tokens, overshoot, optimizer and
  trainer steps, skipped batches, deterministic dataset order/cursor and
  accepted-record digest, base/current/next rollout seeds, parent-process RNG
  identities, config identity, and checkpoint identity.
- The current checkpoint is weights-only. Exact optimizer, cursor, and RNG
  resume is explicitly unsupported.

## Authorization boundary

This CPU stage does not select a token target, alter the existing step-matched
P3 execution manifest, access a model, start serving, run rollout, execute an
optimizer step, or use a GPU. A later outcome-blind target map and a separately
authorized GPU manifest are required before the independent token-budget
schedule can open.

## Validation gate

The stage passes only if targeted config/CLI, backend accounting, engine merge,
terminal ordering, evidence validation, agentic masking, and compatibility
tests pass in a CPU PyTorch environment; Python compilation and
`git diff --check` must also pass. Any excluded or pre-existing suite failure
must be reported separately.
