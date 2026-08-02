# RIST-P2.1-v1.0 terminal report

Formal decision: `INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`

Qwen completed all 256 trajectories and retained all 1,024 raw responses. It
passed every per-model interface and reward-resolution gate: first-turn
executable rate 0.9922, four-turn completion 0.9023, strict success 0.8086,
7/32 mixed groups, 56/256 non-zero-advantage trajectories, and zero retries or
fabricated calls.

The Qwen mixed groups were distributed as low=6 and intermediate=1. Therefore
only one stratum had at least two mixed groups, while the frozen cross-family
gate requires at least two such strata for every checkpoint. The P2.1 PASS
condition is already unreachable regardless of a future Gemma result.

Gemma passed health but failed during the first trajectory with CUDA OOM: the
worker attempted an additional 338 MiB with about 81 MiB free. The server log
records three HTTP 200 scientific responses followed by one HTTP 500, but the
trajectory-transactional client retained zero partial raw responses. This is
both an infrastructure failure and evidence loss. No retry was performed.

The serving window was 1034.774
seconds of the 18,000-second limit. No training occurred, held-out remained
unopened, checkpoints were not downloaded or replaced, and final GPU and
relevant-process listings were empty.

Because the formal stage is invalid, cross-family reward resolution is not
estimable and Qwen cannot be spliced into another revision. Because Qwen also
makes the frozen PASS gate unattainable, an unchanged Gemma-only infrastructure
rerun has no decision value. The current RIST-v1.1 route is closed without
opening P3 training. This does not falsify the broad supervision-topology
hypothesis; it rejects this instrument and execution route.
