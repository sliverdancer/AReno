# RIST-v2.1 main-conference evidence matrix

Current state: the CPU evaluator, exact supervision treatment, frozen external
sources, and Tau3 environment canary pass. Main-conference empirical evidence
has not started.

## Required empirical blocks

| Block | Minimum evidence | Current |
|---|---|---|
| Synthetic causal anchor | resolution bands x AF/LF/AN/LN, powered seeds | C0/D4/P4 contracts frozen; no model outcomes |
| Algorithm replication | GSPO and GRPO | design frozen; no runs |
| Model-family replication | at least 3 checkpoints across 2 families | capacity gate frozen; Qwen serving only, Gemma/24 GB rejected |
| Exact content treatment | full call versus tool-name-only | exact public treatment implemented; checkpoint-tokenizer fixture pending |
| Token robustness | step-matched plus common-token endpoint/AUC | seed-level analyzer, raw-file audit, and blocked 48-run templates frozen |
| Real training environment | Tau3 text airline, preferably retail replication | v1.0.1 isolated; deterministic airline/retail environment canary passed; no training |
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
