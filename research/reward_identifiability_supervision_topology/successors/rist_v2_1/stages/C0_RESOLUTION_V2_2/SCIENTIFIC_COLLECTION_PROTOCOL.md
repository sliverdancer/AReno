# C0 v2.2 scientific collection evidence chain

Status: `CPU_IMPLEMENTED_BLOCKED_UNTIL_CAPACITY_CANARY_PASS`

`build_scientific_collection_manifest.py` is the only admission path into the
scientific pool. It re-runs the capacity finalizer against both raw canary
results and journals, requires an exact outcome-free PASS, and then freezes the
collection source commit, model revisions, tokenizer snapshots, canary GPU UUID,
four jobs of 1,024 trajectories, concurrency eight, and zero retries. A summary
`passed` field alone is not admissible.

`collect_scientific_job.py` rejects a missing canary gate or any runtime identity
mismatch before reading a scientific split. It also re-hashes and replays the
raw canary evidence at job start. Its trajectory artifact and raw response
journal are fresh, immutable inputs to `validate_scientific_job.py`.

The validator checks task/seed identities, turn continuity, exact counts,
runtime identity, and byte hashes without reading or deciding on rewards,
actions, or strict success. `finalize_scientific_collection.py` re-runs all four
validators from raw artifacts and admits exactly 4,096 zero-retry trajectories.
Its PASS opens a separate resolution-analysis stage; collection PASS is not a
scientific result and does not itself reveal or classify any outcome.

No scientific collection manifest is checked in before a real canary PASS.
Consequently, CPU preparation alone cannot accidentally open calibration or
qualification task execution.
