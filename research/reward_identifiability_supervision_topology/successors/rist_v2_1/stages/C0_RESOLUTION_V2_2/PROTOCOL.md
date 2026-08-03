# C0 v2.2 fresh-pool successor protocol

Status: `CPU_ONLY_NOT_GPU_AUTHORIZED`

C0 v2.1 is terminal after Qwen concurrency eight exhausted a 24 GB GPU before
any trajectory or raw response was produced. Its calibration and qualification
tasks are retired and may not be rerun under a lower concurrency.

C0 v2.2 generates new calibration and qualification tasks from a fixed new
seed. The 64 scientific tasks remain balanced across the same eight structural
cells, use new nonces, and are signature-disjoint from the retired C0 pool, the
training split, development split, and the manifest-only sealed confirmatory
split. No held-out task content is generated or opened.

Before any scientific task, both frozen model families must independently pass
an outcome-free capacity canary on a separate hardest-cell task. Each canary
runs eight concurrent, zero-retry, four-turn trajectories with the production
sampling and context configuration. Only request completeness, worker health,
peak GPU memory, and exact model/GPU/source identity may be inspected; task
success is neither reported nor used. Failure of either family retires the
fresh pool without opening calibration.

The successor requires at least 48 GB GPU memory and retains production
concurrency eight for both families. This is a conservative capacity floor, not
an empirical claim that every 48 GB device passes; the canary remains
authoritative. Passing the canary does not authorize C0 collection. New GPU
authorization must separately cover the canary and the four 1,024-trajectory
scientific jobs.

After canary PASS, the original mixed-group estimand, Wilson thresholds,
whole-cell selection, cross-family transport requirement, and final train-pool
filtering remain unchanged. Calibration and qualification use distinct frozen
rollout-seed ranges. Any infrastructure error again retires the v2.2 pool.
