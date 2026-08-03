# X4 Tau3 powered stability expansion

Status: `CPU_FROZEN_AWAITING_X3_PASS_AND_SEPARATE_EXECUTION_AUTHORIZATION`

X4 is opened only after the artifact-backed X3 validator passes all 16 pilot
cells, 3,200 episodes, and 22 airline training tasks per cell. X3 is an
environment-viability pilot and none of its outcomes select X4 seeds, tasks,
arms, checkpoints, algorithms, metrics, or stopping rules.

The independent statistical unit is the paired training seed. A domain-separated
SHA-256 procedure freezes an ordered bank of 32 new seeds before any X3 or X4
outcome is read. The first 12 are a planning floor, not a powered claim. After
all 12 planning seeds finish, their pooled seed-level interaction SD determines
`required_n` under two-sided alpha 0.05, power 0.80, and minimum practical
interaction 0.05. The bank expands in its frozen order to `required_n`; a value
above 32 kills the route. Only `actual_n >= required_n` may be called powered.
The calculation powers one pooled primary interaction; the four
family-algorithm blocks are directional stability gates, not four separately
powered claims. The SD 0.06 used to set the 12-seed floor is planning-only.

Every seed contains the complete Qwen/Gemma, GSPO/GRPO, AF/LF/AN/LN matrix.
The initial tranche has 192 runs; the frozen 32-seed ceiling has 512. Every run
is step-matched at 25 optimizer steps and eight fresh Tau3 episodes per step,
for 200 episodes per run, 38,400 initially, and at most 102,400. All 22 frozen
airline training tasks must occur in every run. Retail, Tau3 test tasks, held-out
data, and BFCL remain inaccessible.

The primary endpoint is the P4-consistent seed-level strict-success interaction
`(AN - AF) - (LN - LF)`, averaged across the four family-algorithm blocks. A GO
requires an absolute mean of at least 0.05, at least 75% paired-seed sign
agreement, no leave-one-seed-out sign reversal, the same nonzero mean sign in
all four blocks, and an observed-SD sensitivity MDE no larger than the observed
effect. Token-normalized AUC and common-support endpoint interactions must keep
the primary sign in every block, and every arm must retain at least 50% token
support. Catastrophic, missing, retried, duplicate, or unbound runs fail closed.

X4 code and manifests are CPU-only templates. They do not authorize a user
simulator, model access, inference, training, GPU use, held-out access, or BFCL.
