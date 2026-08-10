# RIST negative-result handoff

Status: `NEGATIVE_RESULT_POSITIONING_OPEN_GPU_CLOSED`

This directory reframes the terminal RIST v3.1/v4.0 evidence as a negative
result about reward-resolution collapse under group-relative objectives. It is
not a v4 continuation plan and does not authorize any new GPU serving,
qualification, held-out/BFCL access, or training.

Read order:

1. `NEGATIVE_RESULT_POSITIONING.md`
2. `COLLAPSE_TAXONOMY.md`
3. `PAPER_OUTLINE.md`
4. `EVIDENCE_SUMMARY.csv`
5. `EVIDENCE_SUMMARY.json`

The key empirical fact is stable across two independent task-pool lineages:

- `RIST-C0-v3.1`: 2,048 calibration trajectories, zero common high-resolution cells.
- `RIST-C0-v4.0`: 2,048 calibration trajectories, zero common high-resolution cells.

The current admissible next step is CPU-only writing and review: sharpen claims,
add related-work threat matrices, and prepare a negative-result manuscript or
workshop submission. Do not retune v4 after observing the calibration outcomes.
