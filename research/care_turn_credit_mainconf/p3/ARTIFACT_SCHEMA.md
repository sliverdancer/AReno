# P3 artifact contract

Protocol: `CARE-P3-PILOT-v0.1`

## Immutable preparation artifacts

| Path | Required content |
|---|---|
| `manifest.json` | protocol, clean commit/status, source hashes, dataset hash, model asset record, six exact commands, seeds, ceilings |
| `dataset/bifurcation.jsonl` | 64 deterministic task records |
| `GPU_EXECUTION_NOT_AUTHORIZED` | explicit preparation-only marker |
| repository `modelscope_asset.json` | all expected ModelScope file sizes and SHA-256 hashes |

The manifest must be prepared outside the Git checkout so its creation does not
dirty the source tree.

## Raw run artifacts

For each `{arm}-seed-{seed}`:

```text
runs/{run_id}/
├── stdout.log
├── stderr.log
└── metrics/
    ├── events.out.tfevents.*
    └── turn_credit_diagnostics.jsonl
```

`run_results.json` contains `run_id`, arm, seed, completion state, exit code,
wall seconds, log paths, and total single-GPU hours. Failed and timed-out rows
must remain in this file.

## Aligned analysis artifacts

`pilot_steps.csv` and `pilot_steps.json` contain one row per arm, seed, and
trainer step with:

```text
arm
seed
step
reward_mean
reward_std
trainable_tokens
masked_response_tokens
selected_mass
masked_mass
audit_calls
calibration_blocks
update_blocks
valid_trajectories
total_trajectories
threshold
```

The JSON additionally embeds the complete preparation manifest. Collection
fails if either arm or any seed is absent, required TensorBoard tags are
missing, steps are misaligned, diagnostics are empty, thresholds disagree
within a step, or values are non-finite.

Audit calls are trajectory-level quantities repeated on each turn diagnostic;
the collector deduplicates by `trajectory_id` before summation.

## Provenance checks

GPU execution is inadmissible unless:

- the manifest recorded an empty `git status --short`;
- the runtime checkout equals the manifest commit and remains clean;
- the checkpoint is an absolute local directory;
- the asset manifest itself matches its recorded hash;
- every model file matches its frozen size and SHA-256;
- the run set is exactly two arms by three seeds.
