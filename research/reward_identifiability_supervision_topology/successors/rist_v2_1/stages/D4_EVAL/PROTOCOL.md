# D4 independent evaluation split protocol

Status: `CPU_ONLY_FROZEN_BEFORE_MODEL_RESULTS`

D4 uses fresh deterministic split names and nonces from the RIST-v2 generator:

- `dev_curve`: two tasks per structural cell, 16 total; two rollout seeds;
- `confirmatory`: four tasks per structural cell, 32 total; four rollout seeds.

Neither split is used for training. Development curves evaluate checkpoints at
steps 25, 50, 75, and 100. The untrained base checkpoint is evaluated once per
family and reused only as a descriptive baseline. Confirmatory evaluation opens
once after every run, hyperparameter, failure rule, and analysis hash is frozen;
only final step-100 checkpoints are admitted.

The repository materializes only `dev_curve.jsonl`. Confirmatory rows are
committed by deterministic SHA-256 and task signatures in the manifest, then
generated in memory only after the one-shot consumption ledger is durably
created. No confirmatory JSONL is written to the repository.

The strict evaluator exposes dynamic offered tools, requires one call, journals
the raw response before parsing, stops a trajectory at its first invalid action,
performs zero retry/repair, and keeps every trajectory in the denominator.

Training reward is a mechanism diagnostic only. It must never be substituted
for D4 strict task success or sample-efficiency curves.

The parent RIST-v2 D2 held-out was retired after a broad source-search command
opened its file path on 2026-08-02. It is never an input to D4.
