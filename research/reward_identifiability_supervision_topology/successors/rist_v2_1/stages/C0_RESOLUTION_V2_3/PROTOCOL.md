# RIST C0 v2.3 CPU reconstruction protocol

Status: `CPU_RECONSTRUCTION_NOT_GPU_AUTHORIZED`

C0 v2.2 is terminal and may not be repaired or rerun. C0 v2.3 uses a new
capacity-canary task, new calibration and qualification tasks, and new rollout
seed ranges. Its task signatures must be internally unique and disjoint from
all earlier RIST pools, D3 training data, and the manifest-only D4 split.

The deployment contract has one executable entrypoint. The entrypoint must
validate a control-generated immutable envelope before invoking any launcher or
request-capable process. Required bindings are the exact control commit,
runtime commit, manifest SHA-256, model revision, GPU UUID, and compiled
extension SHA-256. Every value is a strict lowercase SHA-256 or exact string as
appropriate; normalization, fallback discovery, and runtime-worktree manifest
lookup are forbidden.

A CPU-only deployment replay uses an injected fake launcher. Each individual
mismatch, the historical wrong-worktree mismatch, and combined mismatches must
exit nonzero before the fake launcher records a call. The held-out merge gate
is separate from development cases. No CPU replay may import a model, start a
server, access a scientific outcome, train, or use a GPU.

Only after the new pool, single entrypoint, replay evaluator, source freeze,
and independent audit all pass may a separate GPU authorization be requested.
