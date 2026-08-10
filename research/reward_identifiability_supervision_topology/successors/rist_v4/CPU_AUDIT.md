# RIST v4 CPU pool freeze audit

Decision: `PASS_CPU_POOL_FREEZE_GPU_CLOSED`

## What passed

- v4 has a separate protocol namespace, directory, seeds, task signatures, and
  result roots.
- The v4 generator is self-owned and does not import v2/v3 generators.
- Calibration and qualification each contain 32 tasks, balanced as 8 cells x 4
  replicates.
- Capacity canary is separate and uses the hardest c07 structure.
- Task signatures and numeric seeds are internally unique and disjoint from
  prior JSON manifests scanned by the builder.
- Codes are short semantic tokens with length 2-3, not hash substrings.
- Prompts do not contain v3 outcome labels, reward-resolution labels, or anchor
  labels.
- Calibration manifest construction does not read qualification content.
- Qualification requires an exact calibration admission.
- Deployment replay rejects wrong worktree/interpreter before child launch.
- Manifest/runtime binding remains non-circular.
- `python -m pytest tests/test_rist_v4_cpu.py -q` passed: 9 passed.

## What this does not prove

No GPU/model request was sent. v4 CPU freeze does not prove that the new pool
will produce common high/low cells. That must be tested later with a separately
authorized A800 calibration canary.

## Gate

- CPU instrument freeze: `GO`.
- GPU readiness: `NO` until explicit user authorization.
- qualification/training/held-out/BFCL: closed.
