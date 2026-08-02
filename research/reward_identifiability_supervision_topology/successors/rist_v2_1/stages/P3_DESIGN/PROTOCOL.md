# RIST-v2.1 P3 factorial design contract

Status: `CPU_DESIGN_ONLY_NOT_EXECUTION_AUTHORIZED`

## Matrix

The minimum pilot is:

- two model families: Qwen3 and Gemma4, each subject to an independent capacity
  and tokenizer gate before execution;
- two algorithms: GSPO and GRPO;
- four arms: AF, LF, AN, LN;
- three paired training seeds per family/algorithm block.

This produces 48 step-matched training runs. No run is currently authorized.
Gemma4 E2B previously failed 24 GB inference capacity, so the matrix does not
claim that a 4090D can train it. A larger-memory GPU canary or a newly frozen
smaller non-Qwen family is required before execution.

## Arm semantics

- AF: all assistant action spans, full call;
- LF: last assistant action span, full call;
- AN: all assistant action spans, argument-masked call;
- LN: last assistant action span, argument-masked call.

AN/LN must not be called `name-only` until checkpoint-tokenizer fixtures prove
that all argument-value tokens are masked and tool-name tokens remain enabled.

## Matching

Step matching is the primary execution budget: every paired block uses the
same initialization, seed, data order, rollout seed derivation, batch shape,
sampling parameters, and optimizer-step count.

Token-matched robustness uses the same runs and a predeclared common cumulative
trainable-token support. The grid depends only on token exposure, never task
success. Report token-indexed endpoints and normalized AUC. This is a
sensitivity analysis, not an independently stopped token-budget training
schedule.

An independent token-budget schedule would require a separately authorized
public `--max-trainable-tokens` API plus optimizer/RNG/cursor-safe stopping and
tests; segmented checkpoint restarts are forbidden because current checkpoints
do not preserve full trainer state.

## Statistical unit and outcomes

Training seed is the replication unit. Tasks and rollouts are repeated
observations within seed. The primary model is a two-sided interaction of
checkpoint-conditional reward resolution with temporal coverage and
call-content eligibility. Required reporting includes strict success,
catastrophic-run rate, sample-efficiency AUC, non-zero-advantage groups,
trainable-token mass, first irrecoverable error, and tool-name/argument
accuracy.

P3 remains diagnostic at three seeds. It may estimate variance and determine
prospective power but cannot establish a main-conference effect.
