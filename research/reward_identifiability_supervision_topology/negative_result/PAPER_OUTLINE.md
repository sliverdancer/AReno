# Paper outline

## Abstract sketch

Multi-turn tool-use reinforcement learning often studies which intermediate
actions should receive supervision. We show that this question has a prior
estimability condition under group-relative objectives: rollout groups must
contain enough terminal reward variation. Across two independently frozen RIST
task-pool lineages, 4,096 calibration trajectories passed infrastructure and
evidence-integrity checks but produced zero common high-resolution cells across
Qwen3-0.6B and Gemma4 E2B. The result demonstrates reward-resolution collapse:
all-fail, all-pass, and non-transportable mixedness can make action-span
supervision effects scientifically unidentifiable even when task oracles and
deployment manifests are valid.

## Structure

### 1. Introduction

- Motivation: supervision topology for multi-turn tool calls is an important
  experimental question.
- Problem: group-relative objectives cannot estimate topology effects without
  reward contrast.
- Contribution: terminal negative evidence from two frozen lineages plus a
  diagnostic taxonomy.

### 2. Background

- Multi-turn tool-use RL.
- Action-span / turn-level supervision.
- Group-relative objectives and non-zero advantage groups.
- Difference between credit assignment and reward-resolution preconditions.

### 3. RIST instrument

- Four-turn tool-call tasks.
- Strict exact tool/argument reward.
- Zero retry.
- Split sealing: calibration before qualification.
- Deployment receipt binding: source commit, manifest SHA, model revision, GPU
  UUID, extension SHA, interpreter identity.

### 4. Terminal lineages

Use `EVIDENCE_SUMMARY.csv` and terminal reports.

| Lineage | Trajectories | Common low cells | Common high cells | Decision |
| --- | ---: | ---: | ---: | --- |
| RIST-C0-v3.1 | 2,048 | 8 | 0 | KILL |
| RIST-C0-v4.0 | 2,048 | 7 | 0 | KILL |

### 5. Collapse taxonomy

- All-fail collapse.
- All-pass collapse.
- Single-model mixedness without transport.
- Copy-code collapse.

### 6. Implications for supervision-topology experiments

- Report reward-resolution diagnostics before training.
- Treat seed/run as statistical units only after reward contrast exists.
- Do not use qualification/held-out splits to tune task pools.
- Separate instrument validity from model capability.

### 7. Limitations

- Only two model families and two RIST lineages.
- No positive training result.
- Synthetic task family only.
- The evidence supports a precondition failure, not a general impossibility
  theorem.

### 8. Next experiments, if pursuing publication

CPU-first:

1. Related-work threat matrix against TRACE, PORTool, TSPO, and other
   turn/step-level credit papers.
2. Formal diagnostic metric: common high/low cell count, mixed-group count,
   non-zero-advantage group rate.
3. External trace audit on public, non-held-out tool-use data if available.

GPU only after a new preregistered external diagnostic plan:

1. Do not mutate v4.
2. Do not open BFCL/held-out as a tuning source.
3. Only run a new benchmark if the generator and GO/KILL rules are frozen
   before model outcomes are inspected.

## Main risk

The paper can be rejected as “benchmark failed to work” unless the diagnostic
contribution is made explicit and backed by related-work positioning. The
stronger framing is:

> reward-resolution checks are a necessary validity layer for group-relative
> multi-turn tool-use RL, analogous to checking non-zero effective sample
> variation before estimating a treatment effect.
