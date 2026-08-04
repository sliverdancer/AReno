# C0 v2.3 calibration terminal record

The frozen calibration supervisor invoked the Qwen job once. The job failed
before creating the deployment ledger because `run_scientific_job.py` called
`python3`, which was absent from the frozen remote `PATH`. Consequently there
was no deployment acceptance, server process, model request, trajectory, or
outcome access. GPU memory returned as 0 MiB and no compute process remained.

This is an infrastructure failure, not scientific evidence about reward
resolution or supervision topology. The preregistered zero-retry rule still
makes the v2.3 calibration protocol terminal. It may not be repaired or rerun;
Gemma, qualification, E1, P3, P4, P5, held-out, and BFCL remain unopened.

The complete receipt, manifest, identity, script, status, and log archive is
stored in the long-term RIST backup directory and is bound by the SHA-256 in
`TERMINAL_RESULT.json`.
