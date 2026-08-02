# RIST-v2.1 main-conference evidence matrix

Current state: CPU evaluator and design contracts only. Main-conference route
is not yet open.

## Required empirical blocks

| Block | Minimum evidence | Current |
|---|---|---|
| Synthetic causal anchor | resolution bands x AF/LF/AN/LN, powered seeds | missing |
| Algorithm replication | GSPO and GRPO | design frozen; no runs |
| Model-family replication | at least 3 checkpoints across 2 families | capacity gate frozen; Qwen serving only, Gemma/24 GB rejected |
| Exact content treatment | full call versus tool-name-only | current argument mask formally insufficient; public change pending |
| Token robustness | step-matched plus common-token endpoint/AUC | analyzer and blocked 48-run templates frozen |
| Real training environment | Tau3 text airline, preferably retail replication | v1.0.1 commit pinned; download absent |
| Sealed external evaluation | BFCL non-live multi-turn categories | v1.3 commit pinned; download and content absent |
| Direct baselines | reward-collapse, turn-credit, and routing/masking neighbors | planned only |
| Confirmatory inference | seed-level CI, prospective power, one-shot held-out | missing |

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
