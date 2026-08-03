# RIST-v2.1 main-conference evidence matrix

Current state: the CPU evaluator, exact supervision treatment, frozen external
sources, and Tau3 environment canary pass. T0b v1.0 reached both frozen models
but terminated on a common serving-metadata defect before multi-turn runtime
qualification. Main-conference empirical evidence has not started.

## Required empirical blocks

| Block | Minimum evidence | Current |
|---|---|---|
| Synthetic causal anchor | resolution bands x AF/LF/AN/LN, powered seeds | C0/D4/P4 contracts frozen; no model outcomes |
| Algorithm replication | GSPO and GRPO | design frozen; no runs |
| Model-family replication | at least 3 checkpoints across 2 families | Qwen and Gemma snapshots verified and each emitted one exact short-canary call on 24 GB; no four-turn runtime or training-capacity qualification; earlier long Gemma trajectory OOM remains relevant |
| Exact content treatment | full call versus tool-name-only | exact public treatment implemented; both tokenizer-only canonical preflights pass; T0b v1.0 terminal on common metadata plumbing; a new runtime fixture remains pending |
| Token robustness | step-matched plus common-token endpoint/AUC | full-interval AUC, endpoint sign, and 50% support gate frozen; no outcomes |
| Real training environment | Tau3 text airline, preferably retail replication | 22-task strict airline adapter CPU-verified; retail blocked by NL-judge confound; no rollout/training |
| Sealed external evaluation | BFCL non-live multi-turn categories | v1.3 code isolated with benchmark data excluded and sealed |
| Direct baselines | reward-collapse, turn-credit, and routing/masking neighbors | planned only |
| Confirmatory inference | seed-level CI, prospective power, one-shot held-out | D4/P4 ledger and analysis contracts frozen; no outcomes |

## GO_MAIN_TRACK

All conditions are necessary:

1. model-conditional resolution transports from calibration to fresh nonces;
2. runtime AF/LF/AN/LN masks are tokenizer-verified and distinct;
3. primary two-sided interaction is at least 10 percentage points in magnitude
   and its 95% interval excludes zero;
4. token-matched sensitivity has no material sign reversal;
5. prospective power and independent training-seed count are met;
6. at least 3 checkpoints across 2 families and both GSPO/GRPO complete;
7. synthetic anchor plus Tau3 real training and BFCL sealed evaluation agree;
8. direct baselines, raw evidence, failure ledger, licenses, and reproducibility
   artifacts are complete;
9. confirmatory held-out opens once with no material protocol deviation;
10. novelty remains prospective resolution-by-topology moderation, not a claim
    of a new static mask or reward-collapse mechanism.

Adequately powered practical-null results, non-transporting calibration,
infeasible seed requirements, real-environment inconsistency, token-match sign
reversal, leakage, or a new direct substitute close the main-track route.

## Current route decision

The paper route remains scientifically open. A 24 GB 4090D is now directly
shown sufficient to load both checkpoints and run the short T0b request, but it
has not passed four-turn qualification or either optimizer-step capacity gate;
the earlier long Gemma OOM still closes Gemma training on that pairing pending
a new E1 result. Publication viability depends first on a separately authorized
additive serving-metadata fix and fresh T0b v1.1, then a larger-memory Gemma E1
pass or a separately frozen non-Qwen replacement. No current evidence supports
upgrading the project to main-conference execution.
