# B2 GPU Inference Qualification

Decision: `INVALID_B2_INFRASTRUCTURE_KV_CACHE_OOM`

The authorized B2 attempt produced no interpretable model completion. Exact
source, data, and Qwen3-0.6B weight hashes passed; the compiled CUDA extension
imported; and nine remote CPU contract tests passed. The failure occurred only
after AReno serving became healthy and received its first inference request.

## Failure

The controller launched AReno with eight running prompts. AReno allocated about
22.5 GiB for model state and KV cache on the 24GB RTX 4090D. The first request
then attempted another 642 MiB allocation while only 557 MiB was free, raising
`torch.OutOfMemoryError`. Rank 0 exited, and every later request returned HTTP
500 with `worker exited without reporting result`.

Four files were written for D128/D512 crossed with seeds 3101/3202. Each file
contains 16 `REQUEST_ERROR` trajectories, so all 64 attempted trajectories lack
a raw model response. Their displayed 0% executability and reward values are
sentinel failure summaries, not scientific observations. N128, N512,
validation, and reserve were not consumed.

The controller should have stopped after the first repeated server failure, but
the client correctly returned zero after preserving the error records and the
controller treated that as a successful cell. The attempt was manually stopped
when this was detected. All serving/controller processes were terminated and
the GPU returned to 0 MiB. A conservative bound from controller start to the
confirmed-idle observation is 125.03 seconds, far below the authorized 3600
seconds.

## Scientific boundary

This attempt says nothing about thinking-budget exhaustion, parser agreement,
four-turn completion, or reward non-degeneracy. It cannot select N128/N512 and
cannot reopen AF/LF training. `SAS-TR-v2.0` is terminal and none of its partial
files may be reused in a later analysis.

The capacity-only successor is frozen in `SAS_TR_V2_1_SUCCESSOR_PLAN.md`. It
must restart all four calibration cells, use one running prompt, add a
single-request preflight and fail-fast checks, and obtain fresh GPU
authorization before execution.
