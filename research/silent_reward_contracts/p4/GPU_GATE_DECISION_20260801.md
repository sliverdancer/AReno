# P4-v0.2 frozen GPU gate decision

Protocol: `ARCA-P4-DYNAMIC-v0.2`

Execution date: `2026-08-01`

Provider: AutoDL

Hardware: one NVIDIA RTX 4090D 24GB

Frozen source commit: `eeb7f73c5efcbd91388b77e99f5f43c391c2e8c2`

Decision: `INVALID_P4_INFRASTRUCTURE_REWARD_IMPORT`

Scientific outcome: `NOT_EVALUATED`

## Answer first

The fail-closed controller launched the first frozen command,
`strict-seed-3101`, and stopped after return code 1. The dynamic reward module
failed to import `examples.agentic.care_bifurcation.task` because the production
reward loader did not expose the repository root on `sys.path`. The failure
occurred before model loading, rollout generation, reward evaluation, or a
training update. The five remaining commands were not opened.

This is an execution-infrastructure invalidation, not evidence for or against
the ARCA reward-contract hypothesis. No repair, retry, selective rerun, or
cross-version result splice is permitted under v0.2.

## Execution evidence

| Field | Value |
|---|---|
| First run | `strict-seed-3101` |
| Return code | `1` |
| Wall time | `12.817546786740422` seconds |
| Single-GPU time | `0.0035604296629834506` hours |
| Model loaded | no |
| Scientific trajectories | `0` |
| Training updates | `0` |
| Remaining runs | `5`, unopened |
| GPU after stop | `0 MiB`, `0%` utilization |

The exact terminal exception was `ModuleNotFoundError: No module named
'examples'` while importing `reward_strict.py` through
`areno.api.rewards.load_reward_fn`.

## Frozen lineage

- manifest SHA-256: `7c53f2dd918ac861cd6a40b313cf974692fbe0d7a35462d5cff6f26aabaa9acf`;
- dataset SHA-256: `bdb6fd0f502a9f7a7bb2e2947a09c89d884ffada7364fb92c81d4105c38fa03d`;
- model-file manifest SHA-256: `a8ca6ec56b3abe7c943eb533b2619063af42f7d1f4527cc9619aa0fc95d2e8ea`;
- run result SHA-256: `0c099c8d225a1aa31d9505ec12a266147e2b21b13d54abbddb0ecf897a09f54d`;
- stderr SHA-256: `3c7072445ca9daead82b3fe370503d87f7fcd8377801ad0fbab3978a48d5b79d`;
- full evidence archive SHA-256: `74f61630a19f924c8419cb613bc3bf7fbb6388ebc0462bc9832ff7073cc83706`.

The full archive is stored under `p4/evidence/`. The prior RIST directory and
evidence were not modified.

## Research consequence

P4 supplies no dynamic scientific result. The current defensible paper remains
the P0-P3 CPU/source audit with explicit mutation-transfer limitations. A new
P4 protocol would require a separate owner decision and must begin from a new
run root; v0.2 may never be resumed.
