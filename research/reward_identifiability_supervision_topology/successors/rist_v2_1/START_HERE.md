# RIST-v2.1 successor

Branch: `research/rist-v2-instrument-reconstruction`

Status: `CPU_ANALYSIS_STACK_PASS_TO_PUBLIC_API_AND_DOWNLOAD_AUTHORIZATION_BOUNDARY`

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
8. X1 pins Tau3 `v1.0.1` and BFCL `v1.3` by exact commit without downloading
   either repository.
9. T0 proves that the existing argument-mask treatment is not yet an exact
   tool-name-only treatment and freezes a fail-closed real-tokenizer gate.
10. E1 freezes per-checkpoint serving plus GSPO/GRPO one-step capacity gates.
11. C0 freezes direct mixed-group resolution calibration and cross-family
    whole-cell selection before training.
12. D4 freezes independent development curves, in-memory confirmatory
    generation after one-shot ledgers, and strict no-repair evaluation.
13. P4/P5 freeze seed-level step interaction, token-common-support robustness,
    prospective power, raw-file verification, and cross-setting transport gates.
14. The parent D2 held-out is permanently retired after a broad source search
    could scan its bytes; it is not an input to any successor result.

## Current boundary

The next scientific step is not training. Two independent authorizations are
now needed before the factorial can become scientifically executable:

1. approve the additive public name-only config/CLI treatment described in
   `stages/T0/PUBLIC_API_CHANGE_REQUEST.md`;
2. approve CPU-only download and isolated installation of the X1-pinned Tau3
   and BFCL sources.

Model/tokenizer access, GPU canaries, the 48-run pilot, and sealed BFCL content
remain separately unauthorized.
