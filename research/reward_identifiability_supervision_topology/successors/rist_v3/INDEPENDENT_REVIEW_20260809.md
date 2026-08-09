# RIST v3 independent CPU replay review

Review date: 2026-08-09

Decision: `PASS_CPU_ONLY_REPLAY_REVIEW_GPU_CLOSED`

## Scope

This was a CPU-only independent review of the v3 collector/finalizer freeze. It
used source inspection, replay-style unit tests, negative corruption tests, and
hash binding. It did not use GPU, model weights, model serving, inference,
training, calibration outcomes, qualification content, held-out tasks, BFCL, or
any v2 outcome values.

## Checks performed

1. Namespace isolation: the scientific collector, validator, and finalizer live
   under `successors/rist_v3/stages/C0_RESOLUTION/` and do not import terminal
   v2.3 collector/finalizer modules or the v2.1 D4 evaluator.
2. Access boundary: the manifest keeps calibration and qualification as separate
   result roots; qualification remains sealed until a later exact admission.
3. Request contract: the collector sends one request per turn, uses the frozen
   sampling contract, zero retry, and emits a raw response journal.
4. Journal integrity: the raw journal is locked during concurrent collection;
   the validator checks per-task/per-seed response counts and contiguous turn
   indices.
5. Terminal finalization: the finalizer refuses to produce final evidence if any
   job validation fails.
6. Negative control: a duplicate task/seed trajectory row is rejected before a
   terminal final collection can be emitted.
7. Replay deployment boundary: wrong-worktree and wrong-interpreter deployment
   receipts still reject before any child process or model request.

## Evidence

- `python -m py_compile` passed for the v3 C0 builders, collector, validator,
  finalizer, and v3 CPU tests.
- `python -m pytest tests/test_rist_v3_cpu.py -q` returned `12 passed`, including frozen calibration manifest and deployment preflight commit-drift negative controls.
- Adjacent v2.3 CPU regressions returned `19 passed, 3 failed`; the failures are
  limited to v2.3 temporary deployment receipt fixture canonicalization and do
  not alter this v3 freeze decision.

## Decision

The v3 CPU scientific request collector and terminal finalizer are frozen for
purposes of building a later clean-commit GPU manifest. GPU/model authority is
still closed. The CPU-side clean-commit calibration manifest and target-host preflight CLI are now ready. The next allowed operation requires explicit GPU authorization to generate the live receipt on the target machine and launch only through the receipt gate.
