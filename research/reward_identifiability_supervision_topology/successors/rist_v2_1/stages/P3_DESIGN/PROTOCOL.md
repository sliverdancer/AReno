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

T0b v1.2 has qualified the production full/name-only masks for both frozen
model families. AN/LN therefore use the public `name_only` treatment; this is a
token-eligibility claim, not a claim that generated arguments are absent.

## Matching

Step matching is the primary execution budget: every paired block uses the
same initialization, seed, data order, rollout seed derivation, batch shape,
sampling parameters, and optimizer-step count.

Token-matched robustness uses the same runs and a predeclared common cumulative
trainable-token support. The grid depends only on token exposure, never task
success. Report token-indexed endpoints and normalized AUC. This is a
sensitivity analysis, not an independently stopped token-budget training
schedule.

The five-point grid is a reporting grid. AUC is integrated on the union of all
observed arm-specific token knots inside common support, so a learning-curve
bend between quartiles is not discarded. Common support must cover at least 50
percent of both each arm's observed evaluation window and its total cumulative
exposure from token zero. Because no outcome is measured before the first
development checkpoint, that left-truncated interval is not imputed into AUC.
The estimand remains retrospective and does not control optimizer-step path,
trajectory count, or token identity.

An independent token-budget schedule would require a separately authorized
public `--max-trainable-tokens` API plus optimizer/RNG/cursor-safe stopping and
tests; segmented checkpoint restarts are forbidden because current checkpoints
do not preserve full trainer state.

The exact scientific boundary and minimum future API are frozen in
`TOKEN_MATCHED_ROBUSTNESS_PLAN.md`; this CPU stage does not change public CLI or
Trainer configuration.

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

## Powered expansion

Before pilot outcomes, P3 freezes an ordered bank of 32 paired seeds. Expansion
is mechanical: require clean transitive evidence, stable same-sign interactions
in all four family/algorithm blocks, no token sign reversal, effect magnitude at
least 0.10, and acceptable catastrophic/zero-advantage rates. The powered design
uses `max(8, maximum prospective requirement across blocks)` seeds and the first
N entries of that bank. A requirement above 32 or any failed pilot gate kills
expansion; seeds or cells may not be selectively replaced. Every expanded run
still requires separate GPU training authorization.
