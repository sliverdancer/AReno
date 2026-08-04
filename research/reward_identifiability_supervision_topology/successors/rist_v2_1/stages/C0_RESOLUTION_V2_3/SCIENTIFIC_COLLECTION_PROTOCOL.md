# C0 v2.3 split scientific collection protocol

Status: `CPU_IMPLEMENTED_GPU_UNEXECUTED`

`build_scientific_collection_manifest.py` is the only admission path. It
requires the frozen v2.3 pool, the exact two-family capacity PASS, a clean
shutdown PASS on the scientific GPU UUID, exact Qwen/Gemma revisions and
tokenizer snapshots, an 80 GB-class GPU, the checked-out source commit, and
zero-retry concurrency eight.

Calibration and qualification are separate two-job manifests. Each contains
Qwen and Gemma over one 32-task by 32-seed grid: 1,024 trajectories per model,
2,048 total. A calibration manifest explicitly denies qualification. A
qualification manifest cannot be built without the exact SHA-256 of a passing
`CALIBRATION_ADMISSION.json`; it explicitly denies calibration.

`collect_scientific_job.py` validates all evidence hashes and runtime identity
before opening the authorized split. `validate_scientific_job.py` checks only
task/seed coverage, hashes, journal continuity, counts, and identity. It does
not branch on rewards or actions. `finalize_scientific_collection.py` replays
both validators and emits a content-blind split receipt.

`run_scientific_job.py` is the only GPU execution entry. It requires control
and runtime roots to resolve to the same clean worktree, launches the exact
receipt through the graceful supervisor, waits for one ledger ACCEPT and the
loopback endpoint, runs one frozen job, then requires clean server exit. The
manifest binds both receipt artifacts; a caller cannot substitute another
server receipt or revive the historical two-worktree ambiguity.

No retry, repair, partial-family substitute, selected-task rerun, held-out,
BFCL access, or training is permitted. A failed or incomplete split is terminal.
