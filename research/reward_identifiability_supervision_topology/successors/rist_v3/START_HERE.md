# RIST v3 handoff

Status: `CPU_V3_1_NONCIRCULAR_PREFLIGHT_READY_GPU_CLOSED`

RIST v3 is a new protocol lineage opened only after
`RIST-R0-REFRESH-v2.0` returned `PASS_NOVELTY_OPEN_STRICT_V3` on 2026-08-05.
It is not a repair, continuation, or selective rerun of terminal C0 v2.3.

Authority order:

1. `INDEPENDENCE_CONTRACT.json` defines the non-reuse boundary.
2. `PROTOCOL.md` defines the scientific and access gates.
3. `stages/C0_RESOLUTION/data/manifest.json` binds fresh tasks and seeds.
4. `stages/C0_RESOLUTION/deployment_entrypoint.py` is the only process-launch
   gate and additionally binds the Python interpreter.
5. `stages/C0_RESOLUTION/collect_scientific_job.py` is the v3-only scientific
   request collector.
6. `stages/C0_RESOLUTION/finalize_scientific_collection.py` is the v3-only
   terminal finalizer; it calls the v3-only validator and refuses invalid jobs.
7. `stages/C0_RESOLUTION/frozen/calibration_stage_manifest_83f9831.json`
   is the frozen calibration stage manifest for commit `83f9831`.
8. `stages/C0_RESOLUTION/bind_manifest_runtime_identities.py` writes
   non-circular static runtime identities into the target-host manifest before
   receipt generation.
9. `stages/C0_RESOLUTION/prepare_deployment_receipt.py` is the target-host
   preflight CLI for producing canonical authority/receipt artifacts before
   serving.
10. `CPU_FREEZE.json`, `CPU_AUDIT.md`, and
   `INDEPENDENT_REVIEW_20260809.md` delimit what is frozen and what still
   blocks a GPU authority.

The v3 lineage, independent pool, access gate, interpreter-bound deployment
foundation, scientific request collector, terminal finalizer, frozen calibration
stage manifest, target-host receipt preflight CLI, and CPU-only independent
replay review are established. No GPU, model, serving, inference, training,
qualification, held-out, or BFCL access is currently open.

Next admissible step requires explicit GPU authorization: on the target machine,
run `prepare_deployment_receipt.py` against the real model snapshot, extension,
Python interpreter, GPU UUID, and frozen calibration manifest; then launch only
through `deployment_entrypoint.py` with the produced receipt SHA.

2026-08-10 correction: v3.1 removes the manifest/receipt circularity. The manifest binds only static runtime identity fields; live runtime evidence carries `deployment_receipt_sha256` after the receipt artifact exists.
