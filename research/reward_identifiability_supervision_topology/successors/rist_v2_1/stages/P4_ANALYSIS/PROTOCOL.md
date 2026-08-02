# P4 frozen seed-level analysis protocol

Status: `CPU_ONLY_FROZEN_BEFORE_RESULTS`

P4 analyzes complete run bundles only. It never discovers arms, drops failed
runs, substitutes task/rollout counts for training seeds, or changes endpoints
after seeing outcomes.

## Statistical units and contrasts

The independent unit is a training seed within a checkpoint-family/algorithm
block. For each paired seed, using AF, LF, AN, and LN:

- temporal main effect: `(AF + AN - LF - LN) / 2`;
- content main effect: `(AN + LN - AF - LF) / 2`;
- primary interaction: `(AN - AF) - (LN - LF)`.

Catastrophic runs remain in the intention-to-treat endpoint with strict success
zero. A missing run, missing raw-evidence hash, or unaligned arm makes the block
invalid rather than being omitted.

Every raw-evidence hash is recomputed from an explicitly rooted relative file.
A syntactically valid digest without the corresponding file cannot pass. The
analyzer also refuses main-track eligibility when the manifest is still a
template, lacks execution authorization, or contains any treatment not marked
scientifically ready.

## Step- and token-matched estimands

The primary endpoint is one-shot D4 confirmatory strict success at the common
frozen optimizer-step budget. Development checkpoint success supplies the
learning curve only. Token robustness combines that development curve with the
same runs' outcome-blind common cumulative-trainable-token support from P3. It
reports normalized AUC and its same paired interaction. A token curve with fewer
than two valid observations or no common support is not estimable and blocks
the main route.

Resolution-band endpoints are reported separately for mechanism diagnosis.
Their direction may be learned in the three-seed pilot, but any confirmatory
moderation direction and threshold must be frozen before additional seeds.

## Stability and power gates

Three seeds are pilot-only. Main-track eligibility requires, within all four
family/algorithm blocks:

1. at least `max(8, prospective_power_requirement)` paired seeds;
2. absolute mean step interaction at least 0.10;
3. deterministic seed-bootstrap 95 percent interval excluding zero;
4. the same nonzero sign across all four blocks;
5. token-AUC interaction with the same sign and no block sign reversal;
6. catastrophic-run and zero-advantage-run rates each at most 0.10;
7. complete raw evidence and no post-outcome exclusions.

The bootstrap interval is descriptive when seed count is small. It does not
override the prospective power gate.
