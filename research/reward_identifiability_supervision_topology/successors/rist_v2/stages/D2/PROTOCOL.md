# RIST-v2 D2 structural candidate-pool freeze

Protocol: `RIST-D2-v2.0`

Mode: CPU-only generation and structural validation

## Purpose

Create independent tasks for model calibration, prospective qualification, and
sealed held-out evaluation without assigning model-free reward-resolution
labels. D2 freezes what may later be calibrated; it does not select cells and
does not access a model.

## Structural cells

Eight predeclared cells cross controlled amounts of:

- semantically adjacent distractor tools;
- argument-record decoys;
- cross-turn dependency depth;
- direct versus relational candidate selection.

Every task retains exactly four ordered actions and one unique strict oracle.
Each split contains four independent nonces per cell:

- calibration: 32 tasks;
- qualification: 32 tasks;
- held-out: 32 sealed tasks.

The later calibration process may retain or reject entire structural cells. It
may not select individual tasks based on outcomes. Qualification uses fresh
nonces to test transport. Held-out is generated and hash-bound but remains
unopened by selection and qualification code.

## CPU gates

1. byte-identical regeneration;
2. 8 balanced cells and 32 tasks per split;
3. unique, cross-split-disjoint signatures;
4. exactly four oracle actions and a unique candidate/tool choice per turn;
5. all requested difficulty dimensions vary;
6. no model-derived or analytic reward-resolution label is stored;
7. selection code has no held-out input;
8. GPU, serving, inference, training, download, and checkpoint access remain
   false.

Pass returns `PASS_D2_CPU_POOL_TO_GPU_AUTHORIZATION_REQUEST`. It opens only a
future protocol-design boundary; it does not authorize E0 or model calibration.
