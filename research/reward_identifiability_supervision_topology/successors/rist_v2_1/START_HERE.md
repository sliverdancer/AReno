# RIST-v2.1 successor

Branch: `research/rist-v2-instrument-reconstruction`

Status: `D2_1_PREFREEZE_ARBOR_SEARCH`

Parent terminal result: `../rist_v2/stages/D2/stage_result.json`

## Scope

This is a new protocol revision. It does not alter or rerun terminal RIST-v2
D2. The archived parent score is 7/8. D2.1 uses a new split-local evaluator to
improve that engineering contract without changing the generator, data,
thresholds, or scientific hypotheses.

Only calibration may guide Arbor search. Qualification is a one-shot merge
gate. Held-out is neither a path nor an input to the evaluator and must not be
generated, opened, copied, or parsed.

GPU, model access, downloads, inference, and training remain closed.

## Required order

1. Record archived M0 score 7/8 at commit `49b09ec`.
2. Run HTR candidates on calibration only.
3. Select one candidate mechanically by dev score; qualification cannot break
   a tie.
4. Consume qualification once for the selected candidate.
5. Merge only if score is 8/8 and all integrity gates pass.
6. Freeze D2.1, then prepare later CPU-only model/training/environment
   contracts. Stop before downloads or GPU use.
