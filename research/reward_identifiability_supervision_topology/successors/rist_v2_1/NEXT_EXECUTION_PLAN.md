# RIST-v2.1 outcome execution plan

Status: `C0_V2_3_P0_CPU_FREEZE_PASS_GPU_OFFLINE`

The objective is not satisfied by a passing tokenizer fixture or a short
serving check. It requires observed training-seed outcomes for the synthetic
instrument and Tau3. Every stage below is fail-closed and keeps discovery,
calibration, development curves, and confirmatory evaluation separate.

## Ordered gates

1. **T0b v1.2 treatment qualification — passed.** On the existing locked Qwen3-0.6B and
   Gemma4 E2B snapshots, import and hash the compiled extension before serving,
   then collect exactly 32 fresh four-turn rows per model with zero retry. Build
   and evaluate the production masks separately per model. Both must pass.
2. **C0 v2.2 group-resolution calibration — terminal before scientific access.** C0 v2.1
   and its tasks are terminal after the 24 GB concurrency failure. On a newly
   bound GPU with at least 48 GB, first run the outcome-free two-family capacity
   canary at concurrency eight. Only an exact raw-evidence PASS opens the fresh,
   signature-disjoint four-job, 4,096-trajectory calibration/qualification
   pool. Select only whole transported cells; any canary, infrastructure, or
   common-map failure kills v2.2 without repair or selective rerun. The
   post-collection resolution replay, cross-family transport, filtered D3
   output, independent validation receipt, and pre-access root are now frozen
   by `RESOLUTION_CPU_FREEZE.json` and `RESOLUTION_EXECUTION_ROOT.json`. On the
   A800 execution, the Qwen outcome-free requests completed, but their
   pre-request identity bound superseded source commit `2674e101` instead of
   required runtime commit `f00b688d`. Fail-closed zero-retry rules consume and
   kill v2.2; Gemma and all scientific splits remained unopened.
3. **C0 v2.3 split calibration/qualification — CPU-frozen, GPU unexecuted.**
   A fresh disjoint pool and a single fail-closed deployment entrypoint passed
   two-family capacity and clean-shutdown canaries. Scientific execution now
   collects Qwen and Gemma calibration only (2,048 trajectories), freezes a
   common whole-cell map, and constructs the separate 2,048-trajectory
   qualification manifest only after calibration GO. Both stages use
   concurrency eight and zero retry. A GPU UUID change requires fresh binding
   and outcome-free clean-shutdown admission. The prior A800 endpoint is
   currently unreachable; no scientific outcome has been opened.
4. **E1 capacity — admission CPU-frozen, authorized, unexecuted.** The v2.3
   admission builder accepts only `PASS_C0_V2_3_TO_E1_CAPACITY`, retains whole
   D3 cells, and selects exactly four tasks from the lexicographically first
   transported high cell. On the deployment-bound GPU, run
   serving plus exactly one optimizer step for GSPO and GRPO per checkpoint.
   Qwen and Gemma remain training-unqualified; Gemma requires at least 48 GB and
   15% measured headroom. The released 24 GB instance is no longer available
   and cannot be treated as a qualified pairing. E1 starts only after C0 creates
   the transported high-resolution AF canary dataset. The v2.2 deployment
   builder is frozen by `V2_2_DEPLOYMENT_CPU_FREEZE.json`; it cannot emit an
   executable manifest without the reproduced C0 validation receipt, existing
   E1 authorization, exact model revisions, the capacity-data hash, and an
   actual same-GPU runtime identity with at least 48 GiB. Reserve an 80 GB GPU
   because 48 GB is only the hard canary floor for Gemma.
5. **P3 diagnostic pilot.** Execute all AF/LF/AN/LN arms for both families,
   both algorithms, and three paired seeds: 48 step-matched runs. No partial
   family or algorithm block can stand in for the full objective.
6. **P4 decision and power.** Use training seed as the independent unit. Require
   at least 75% seed-sign agreement, no leave-one-seed-out sign reversal,
   cross-block direction agreement, token-AUC and token-endpoint direction
   preservation, acceptable catastrophic/zero-advantage rates, and prospective
   seed power. Three seeds estimate variance only.
7. **Tau3 airline external validity.** Freeze one user-simulator revision and
   run the clean-reset 25-step one-seed factorial airline pilot: 16 runs and
   3,200 episodes, with zero policy and user retries. Only after a viable pilot
   may the same family/algorithm/arm design scale to powered seeds. Retail
   remains blocked by the NL-judge confound.
8. **M0-M6 third-checkpoint replication.** The outcome-blind M0 rule now
   preregisters Qwen3-1.7B, using the existing dense Qwen3 adapter and no new
   dependency. It remains unqualified and contributes no evidence. Only after
   the original two-checkpoint P3/P4 gate stays open may separately authorized
   tokenizer/config, exact weights, runtime treatment, C0 transport, E1, and
   powered GSPO/GRPO AF/LF/AN/LN replication be run. Scientific failure forbids
   switching checkpoints; only a pre-inference metadata/license failure may
   invoke the preregistered Qwen3-4B fallback.
9. **Confirmation and paper gate.** Open the one-shot confirmatory split and
   sealed BFCL only after all earlier gates pass. Main-conference upgrade also
   requires the qualified third checkpoint, direct baselines, and a complete
   raw-evidence/license/reproduction archive.

## Current authorization boundary

Step 1 passed with 32 runtime rows and 32 exact masks per model. The A800 was
correctly bound and both locked model snapshots passed their historical file
locks, but C0 v2.2 is terminal because the first Qwen canary used a superseded
control manifest to build its runtime identity. Post-hoc correction and rerun
are prohibited. The fresh v2.3 successor used new tasks and passed the
non-scientific A800 capacity and clean-shutdown gates. Its split scientific
protocol and CPU freeze now pass, but the last A800 SSH endpoint is offline. No
calibration/qualification outcome, E1 training, held-out/BFCL, Tau3 policy, P3,
or M1-M6 access has occurred. Standing GPU authorization does not bypass the
ordered GO/KILL gates.
