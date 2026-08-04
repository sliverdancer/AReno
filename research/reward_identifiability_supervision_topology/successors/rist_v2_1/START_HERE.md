# RIST-v2.1 successor

Branch: `research/rist-v2-instrument-reconstruction`

Status: `C0_V2_3_P0_CPU_FREEZE_PASS_GPU_OFFLINE`

Parent terminal result: `../rist_v2/stages/D2/stage_result.json`

## Scope

This is a new protocol revision. It does not alter or rerun terminal RIST-v2
D2. The archived parent score is 7/8. D2.1 uses a new split-local evaluator to
improve that engineering contract without changing the generator, data,
thresholds, or scientific hypotheses.

Only calibration may guide Arbor search. Qualification is a one-shot merge
gate. Held-out is neither a path nor an input to the evaluator and must not be
generated, opened, copied, or parsed.

The separately authorized T0b model downloads and inference are complete.
The user has authorized the ordered GPU inference/training route through P5.
Execution remains mechanically gated: failure closes downstream stages, and
held-out/BFCL remain sealed until their preregistered one-shot gates.

## Required order

1. Archived M0 remained 7/8 at commit `49b09ec`.
2. H1 and H2 both scored 8/8 on calibration.
3. The frozen tie rule selected H1 before qualification.
4. H1 scored 8/8 on the one-shot qualification merge gate.
5. H1 was merged as Arbor `M_best`; no post-test search occurred.
6. D3 generated a balanced, disjoint 32-task train split and froze a strict
   no-repair dynamic-tool runner plus exact binary reward.
7. P3 and external-environment CPU contracts are frozen.
8. X1 acquired and isolated Tau3 `v1.0.1` and BFCL `v1.3` at exact commits;
   BFCL benchmark data was excluded and remains sealed.
9. The public `--tool-call-supervision {full,name_only}` treatment is
   implemented with a strict offset-mapped, fail-closed mask. T0a downloaded
   immutable tokenizer-only snapshots and passed the 32-case canonical
   preflight for both Qwen3-0.6B and Gemma4 E2B. T0b v1.0 then terminated after
   one exact call per model because the common HTTP response schema removed the
   required runtime token IDs. The consumed v1.0 result is an infrastructure
   failure and does not qualify the treatment. T0b v1.1 then collected all 32
   Gemma rows but failed the two-family gate after a missing-extension Qwen
   deployment and exposed that the frozen mask recognized JSON calls but not
   Gemma tokenizer-native call syntax. The consumed result remains engineering
   evidence only. T0b v1.2 freezes fresh inputs, the corrected strict mask, and
   a mandatory compiled-extension import/hash preflight. T0b v1.2 then passed
   both families: 32 fresh four-turn rows and 32 exact production masks per
   model with zero retry.
10. E1 freezes per-checkpoint serving plus GSPO/GRPO one-step capacity gates.
11. C0 freezes direct mixed-group resolution calibration and cross-family
    whole-cell selection before training.
12. D4 freezes independent development curves, in-memory confirmatory
    generation after one-shot ledgers, and strict no-repair evaluation.
13. P4/P5 freeze seed-level step interaction, token-common-support robustness,
    prospective power, raw-file verification, and cross-setting transport gates.
14. The parent D2 held-out is permanently retired after a broad source search
    could scan its bytes; it is not an input to any successor result.
15. X2.0 permanently killed the independently-executable gold-action replay
    assumption. The new upstream-aligned X2.1 passed two 28-test runs and two
    exact clean-reset mutating canaries without opening the Tau3 task test split.
16. P4 token robustness now covers the entire common token interval, rejects
    less than 50 percent support in any arm, and requires both AUC and endpoint
    sign preservation.
17. X3 provides a CPU-verified Tau3 airline training adapter over 22 strict
    `DB x COMMUNICATE` tasks. All 66 retail tasks are blocked domain-wide because
    65 invoke `NL_ASSERTION` and would add an LLM judge to the reward.

## Current boundary

T0b v1.0 is terminal at `stages/T0B/gpu_run_20260803/FINAL_RESULT.json`.
Both exact snapshots loaded and both models emitted the required first exact
tool call, but neither HTTP response retained `areno.response_tokens`. A
CPU-only inspection localized two faults: serve never enables the shared
builder's metadata flag, and the Pydantic envelope would independently drop the
extension. The result has no scientific interpretation and must not be repaired
or rerun.

The additive public serving-response metadata fix passes CPU regression tests at
commit `b6d7bdc`; v1.1 is terminal at commit `3de4e0c`; and T0b v1.2 passed from
the frozen `3c28f06` deployment. The 24 GB 4090D completed short four-turn
serving for both families, but the earlier long Gemma4 trajectory OOM still
means E1 capacity remains unresolved. The fresh C0 v2.3 pool, exact deployment
gate, A800 capacity canary, and clean-shutdown canary passed without scientific
access. Its scientific protocol now separates 2,048 calibration trajectories
from 2,048 qualification trajectories; qualification cannot be constructed
until calibration freezes at least two common low and two common high whole
cells. The CPU freeze passes and no scientific outcome has been opened. The
last A800 endpoint is offline, so calibration awaits a live 80 GB GPU, fresh
UUID binding, and outcome-free clean-shutdown admission. E1 and training remain
unopened by the C0 gate despite their standing authorization.
