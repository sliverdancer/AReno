# RIST-v2.1 successor

Branch: `research/rist-v2-instrument-reconstruction`

Status: `D3_PASS_CPU_INSTRUMENT_TO_DOWNLOAD_AUTHORIZATION_BOUNDARY`

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

1. Archived M0 remained 7/8 at commit `49b09ec`.
2. H1 and H2 both scored 8/8 on calibration.
3. The frozen tie rule selected H1 before qualification.
4. H1 scored 8/8 on the one-shot qualification merge gate.
5. H1 was merged as Arbor `M_best`; no post-test search occurred.
6. D3 generated a balanced, disjoint 32-task train split and froze a strict
   no-repair dynamic-tool runner plus exact binary reward.
7. P3 and external-environment CPU contracts are frozen. Stop before benchmark
   downloads, environment installation, model access, or GPU use.

## Current boundary

The next scientific step is not training. It is a separately authorized CPU
download/audit of pinned Tau3 and BFCL sources, followed by a model-capacity
and tokenizer qualification protocol. The 48-run training matrix remains
unauthorized.
