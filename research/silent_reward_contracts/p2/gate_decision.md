# P2 gate decision

Decision: `PASS_P2_CROSS_SYSTEM_TO_P3`

Date: `2026-08-01`

The frozen structured artifact reports:

- natural medium/high classes detected: `3`;
- distinct failure classes: `3`;
- independent systems: `2`;
- registered conclusion/gate flips: `2`.

This meets the exact P2 prevalence gate. P3 is opened once, using OpenRLHF at
the preregistered held-out commit. The auditor rules and thresholds must be
frozen before reading its source. P4 remains unopened and requires separate GPU
authorization even if P3 passes.
