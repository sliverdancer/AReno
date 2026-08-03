# C0 authorization boundary

Status: `USER_AUTHORIZED_AWAITING_GPU_ENDPOINT_AND_BINDING_FREEZE`

C0 requires fresh GPU inference against the exact Qwen3-0.6B and Gemma4 E2B
revisions. It performs 4,096 four-turn trajectories (up to 16,384 response
calls), zero retry, without training. The observed T0b latency gives an ideal
concurrent estimate of 3.31 GPU-hours and a planning estimate of 4.47 hours;
because the frozen Qwen-8/Gemma-4 concurrency has not yet been validated under
this workload, request an 8-hour ceiling.

The user authorized model serving and C0 collection on 2026-08-03. The
repository manifest remains non-executable until the exact source commit,
model paths, GPU UUID, output paths, and authorization hash are bound during
deployment. The grant does not silently include P3 training, Tau3,
held-out/BFCL, model replacement, or retry. Calibration and qualification run
under one server launch per model, Qwen first and Gemma second. A failure
consumes that family/split job; do not repair or selectively rerun it.
