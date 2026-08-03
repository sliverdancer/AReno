# RIST-v2.1 outcome execution plan

Status: `T0B_V1_2_PASS_AT_C0_AND_E1_AUTHORIZATION_BOUNDARY`

The objective is not satisfied by a passing tokenizer fixture or a short
serving check. It requires observed training-seed outcomes for the synthetic
instrument and Tau3. Every stage below is fail-closed and keeps discovery,
calibration, development curves, and confirmatory evaluation separate.

## Ordered gates

1. **T0b v1.2 treatment qualification — passed.** On the existing locked Qwen3-0.6B and
   Gemma4 E2B snapshots, import and hash the compiled extension before serving,
   then collect exactly 32 fresh four-turn rows per model with zero retry. Build
   and evaluate the production masks separately per model. Both must pass.
2. **C0 group-resolution calibration.** Run the frozen four-job, 4,096-trajectory
   calibration/qualification manifest. Select only whole structural cells whose
   collapsed/resolved label transports to fresh nonces and intersects across
   both families. A failed common map kills the current synthetic pool.
3. **E1 capacity.** For each checkpoint/GPU pairing, run serving plus one
   optimizer step for GSPO and GRPO. Qwen remains unqualified on 24 GB; Gemma is
   already rejected on 24 GB. Gemma therefore needs a fresh larger-memory
   canary or a separately versioned and fully qualified non-Qwen replacement.
4. **P3 diagnostic pilot.** Execute all AF/LF/AN/LN arms for both families,
   both algorithms, and three paired seeds: 48 step-matched runs. No partial
   family or algorithm block can stand in for the full objective.
5. **P4 decision and power.** Use training seed as the independent unit. Require
   at least 75% seed-sign agreement, no leave-one-seed-out sign reversal,
   cross-block direction agreement, token-AUC and token-endpoint direction
   preservation, acceptable catastrophic/zero-advantage rates, and prospective
   seed power. Three seeds estimate variance only.
6. **Tau3 airline external validity.** Freeze one user-simulator revision and
   run the clean-reset 22-task stateful airline pilot with zero policy and user
   retries. Only after a stable pilot may the same family/algorithm/arm design
   scale to powered seeds. Retail remains blocked by the NL-judge confound.
7. **Confirmation and paper gate.** Open the one-shot confirmatory split and
   sealed BFCL only after all earlier gates pass. Main-conference upgrade also
   requires a third checkpoint across the two families, direct baselines, and a
   complete raw-evidence/license/reproduction archive.

## Current authorization boundary

Step 1 passed with 32 runtime rows and 32 exact masks per model. Steps 2 and 3
are the next independent gates and retain `execution_authorized=false`. The T0b
grant does not cover C0 model rollout, optimizer steps, training, Tau3
user-simulator calls, held-out data, or BFCL content.
