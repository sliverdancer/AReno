# P2 Cross-Family Inference Qualification

Protocol: `RIST-P2-v1.0`

Status: `FROZEN_AWAITING_GPU_AUTHORIZATION`

Frozen: `2026-08-01`

## Purpose

Test whether the CPU analytic reward-resolution range is realized by actual
language models before opening any supervision-topology training. P2 is
inference only and cannot estimate AF/LF/AN/LN efficacy.

## Checkpoints

Run sequentially on one RTX 4090D:

1. `Qwen/Qwen3-0.6B` via the existing Qwen3 adapter;
2. `google/gemma-4-E2B-it` via the existing Gemma4 adapter.

These are separate model families. Exact config, tokenizer, weight-file, and
combined snapshot hashes must be recorded after download and before serving.
If the Gemma4 checkpoint cannot load through the reviewed adapter, return
`INVALID_P2_MODEL_ADAPTER`; do not substitute a third checkpoint after seeing
Qwen outcomes.

## Data and sampling

- data: P1 qualification only, 32 tasks;
- held-out: unopened;
- rollout seeds: `6101, 6202, 6303, 6404, 6505, 6606, 6707, 6808`;
- trajectories: 256 per checkpoint, 512 total;
- action opportunities: exactly four per trajectory;
- thinking: disabled;
- maximum new tokens: 128 per action;
- temperature: 1.0;
- top-p: 1.0;
- maximum running prompts: 1;
- no retries after a scientific response;
- no call repair, synthesis, or parser fallback.

For `forced` cells, expose only the expected tool and force that name. For
`free` cells, expose the expected tool plus frozen distractors and require one
of the offered tools without naming which one. Every tool uses the task's
explicit `code` enum. A wrong name or code returns a deterministic failure
observation and never reveals the hidden next code. All four responses remain
in the raw evidence even after strict success becomes impossible.

## Primary qualification measures

Per checkpoint and task:

- first-turn executable-call rate;
- four-turn parsed-call completion rate;
- strict exact-oracle success;
- unique full action sequences across eight seeds;
- mixed strict-reward group indicator;
- non-zero group-relative-advantage trajectory count;
- empirical reward entropy and advantage-collapse rate;
- raw/parsed agreement and invalid-reason counts.

## Frozen gates

Each checkpoint must satisfy:

1. first-turn executable rate at least `0.90`;
2. four-turn parsed-call completion rate at least `0.75`;
3. overall strict success in `[0.05, 0.95]`;
4. at least `6/32` mixed strict-reward task groups;
5. at least `48/256` non-zero-advantage trajectories;
6. zero fabricated/repaired calls and complete raw evidence.

The cross-family instrument must additionally satisfy:

1. both checkpoints pass the interface and signal gates;
2. at least two P1 analytic strata each contain at least two mixed groups per
   checkpoint;
3. at least one task per checkpoint has multiple action sequences but a
   homogeneous strict reward, preserving the action/reward distinction;
4. qualification outputs are not used to change tasks, sampling, thresholds,
   or checkpoint identities.

## Outcomes

- `PASS_P2_CROSS_FAMILY_RESOLUTION_TO_P3_PILOT`
- `KILL_TASK_INSTRUMENT_NO_MODEL_RESOLUTION`
- `KILL_CHECKPOINT_INTERFACE`
- `INVALID_P2_MODEL_ADAPTER`
- `INVALID_P2_INFRASTRUCTURE`

Any `KILL` closes the current route. An infrastructure or adapter invalidity
does not support a scientific claim and cannot be spliced with a replacement
model without a new protocol.

## Resource and authorization boundary

Requested ceiling: at most two GPU-hours total for sequential serving and 512
trajectories. Download time is not GPU time. Stop immediately on OOM, non-finite
output, evidence loss, model-hash drift, repeated server failure, or timeout.
No optimizer, checkpoint save, or training command is permitted in P2.

