# P5 cross-setting main-track gate

Status: `CPU_ONLY_FROZEN_BEFORE_RESULTS`

P5 is evaluated only after independently frozen analyses exist for:

1. RIST-v2.1 synthetic causal anchor;
2. Tau3 stateful real-tool environment;
3. sealed BFCL non-live multi-turn transfer.

Every setting must use both model families, both algorithms, complete raw
evidence, seed-level uncertainty, and the same treatment contrast. Synthetic
and Tau3 must additionally pass token-common-support robustness. BFCL is opened
once only after training and hyperparameters are frozen.

The main route passes only when every setting is complete, every 95 percent
interval excludes zero, interaction signs agree, no token sign reversal occurs,
and catastrophic-run rates stay at or below 0.10. Missing or invalid evidence
is not a null result; it is `BLOCKED_OR_INVALID`.

Every setting additionally needs a prospective power pass and the hash of its
pre-result analysis protocol. Tau3 must carry a passed deterministic environment
qualification; BFCL must attest that sealed content was consumed exactly once.
