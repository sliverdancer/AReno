# RIST-v2.1 outcome execution plan

Status: `C0_V2_2_AND_E1_AUTHORIZED_AWAITING_NEW_GPU_BINDING`

The objective is not satisfied by a passing tokenizer fixture or a short
serving check. It requires observed training-seed outcomes for the synthetic
instrument and Tau3. Every stage below is fail-closed and keeps discovery,
calibration, development curves, and confirmatory evaluation separate.

## Ordered gates

1. **T0b v1.2 treatment qualification — passed.** On the existing locked Qwen3-0.6B and
   Gemma4 E2B snapshots, import and hash the compiled extension before serving,
   then collect exactly 32 fresh four-turn rows per model with zero retry. Build
   and evaluate the production masks separately per model. Both must pass.
2. **C0 v2.2 group-resolution calibration — authorized, unexecuted.** C0 v2.1
   and its tasks are terminal after the 24 GB concurrency failure. On a newly
   bound GPU with at least 48 GB, first run the outcome-free two-family capacity
   canary at concurrency eight. Only an exact raw-evidence PASS opens the fresh,
   signature-disjoint four-job, 4,096-trajectory calibration/qualification
   pool. Select only whole transported cells; any canary, infrastructure, or
   common-map failure kills v2.2 without repair or selective rerun.
3. **E1 capacity — authorized, unexecuted.** On the deployment-bound GPU, run
   serving plus exactly one optimizer step for GSPO and GRPO per checkpoint.
   Qwen and Gemma remain training-unqualified; Gemma requires at least 48 GB and
   15% measured headroom. The released 24 GB instance is no longer available
   and cannot be treated as a qualified pairing. E1 starts only after C0 creates
   the transported high-resolution AF canary dataset.
4. **P3 diagnostic pilot.** Execute all AF/LF/AN/LN arms for both families,
   both algorithms, and three paired seeds: 48 step-matched runs. No partial
   family or algorithm block can stand in for the full objective.
5. **P4 decision and power.** Use training seed as the independent unit. Require
   at least 75% seed-sign agreement, no leave-one-seed-out sign reversal,
   cross-block direction agreement, token-AUC and token-endpoint direction
   preservation, acceptable catastrophic/zero-advantage rates, and prospective
   seed power. Three seeds estimate variance only.
6. **Tau3 airline external validity.** Freeze one user-simulator revision and
   run the clean-reset 25-step one-seed factorial airline pilot: 16 runs and
   3,200 episodes, with zero policy and user retries. Only after a viable pilot
   may the same family/algorithm/arm design scale to powered seeds. Retail
   remains blocked by the NL-judge confound.
7. **M0-M6 third-checkpoint replication.** The outcome-blind M0 rule now
   preregisters Qwen3-1.7B, using the existing dense Qwen3 adapter and no new
   dependency. It remains unqualified and contributes no evidence. Only after
   the original two-checkpoint P3/P4 gate stays open may separately authorized
   tokenizer/config, exact weights, runtime treatment, C0 transport, E1, and
   powered GSPO/GRPO AF/LF/AN/LN replication be run. Scientific failure forbids
   switching checkpoints; only a pre-inference metadata/license failure may
   invoke the preregistered Qwen3-4B fallback.
8. **Confirmation and paper gate.** Open the one-shot confirmatory split and
   sealed BFCL only after all earlier gates pass. Main-conference upgrade also
   requires the qualified third checkpoint, direct baselines, and a complete
   raw-evidence/license/reproduction archive.

## Current authorization boundary

Step 1 passed with 32 runtime rows and 32 exact masks per model. C0 v2.2 and E1
are explicitly GPU-authorized, but the freeze remains `deployment_bound=false`:
the previous 24 GB instance was released and no current endpoint/GPU UUID is
bound. Therefore no GPU command is executable yet. Tau3 user-simulator calls,
P3 training, held-out/BFCL access, and every M1-M6 third-checkpoint model access
remain unauthorized and require separate gates.
