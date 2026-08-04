# RIST C0 v2.3 CPU reconstruction protocol

Status: `P0_CPU_FREEZE_PASS_GPU_OFFLINE`

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

The capacity and clean-shutdown gates passed on the A800 without scientific
access. Scientific execution is split into two irreversible stages. Calibration
collects exactly 2,048 trajectories across both families and cannot read the
qualification task file. Its outcome analysis freezes a common whole-cell map.
Only an exact `PASS_CALIBRATION_TO_QUALIFICATION` admission can construct the
separate 2,048-trajectory qualification manifest. Qualification may confirm or
reject the frozen cells but cannot add a cell that calibration did not select.

Both stages use concurrency eight and zero retry. Infrastructure failure,
identity mismatch, evidence tampering, or a post-access exception consumes the
stage and produces a terminal KILL; repair, selective rerun, and result-driven
threshold changes are forbidden. Held-out, BFCL, and training remain closed.

The frozen resolution rule sorts the 32 seeds and divides them into four groups
of eight per task. A group is mixed when it contains both strict rewards. Each
four-task cell has 16 groups. A cell is collapsed when its 95% Wilson upper
bound is at most 0.25, resolved when its lower bound is at least 0.50, and
transition otherwise. A collapsed or resolved cell also requires at least three
of four tasks to agree; otherwise it is heterogeneous. Calibration and
qualification each require at least two common low and two common high cells.

Only after the source freeze, CPU regressions, deployment replay, access-boundary
tests, and independent audit pass may the already granted GPU authorization be
used. A GPU UUID change invalidates the prior clean-shutdown receipt and requires
a fresh outcome-free binding/canary before calibration.
