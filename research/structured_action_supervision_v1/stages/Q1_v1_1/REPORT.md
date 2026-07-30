# Q1 v1.1 Bounded GPU Integration Pilot

Decision: `KILL_CURRENT_INSTRUMENT`

The engineering successor succeeded: the Python 3.12 dynamic-loader defect was
repaired, 156 targeted tests passed on the rented machine, and all four frozen
Q1 cells completed with return code 0. The scientific instrument nevertheless
failed its prespecified qualification gates. Across 128 raw trajectories,
Qwen3-0.6B produced zero strictly executable four-call trajectories; every
trajectory ended with `MISSING_TOOL_CALL` at the first assistant response.

This result cannot estimate the effect of all-assistant-turn supervision (AF)
versus last-assistant-turn supervision (LF).

## Frozen execution

- protocol: `SAS-P0-v1.1`;
- source: commit `f96c96544284d65bcb2e8e10ab1d9f2aeda08cd6`;
- model: Qwen3-0.6B, ModelScope weights SHA256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
- GPU: NVIDIA GeForce RTX 4090 D;
- cells: AF/LF crossed with seeds 1101/2202;
- eight optimizer steps and four samples per step in every cell;
- controller time: 410.19 seconds;
- held-out outcomes were not consumed.

All four commands completed without OOM or non-finite TensorBoard scalars.
Complete logs, run configs, TensorBoard events, and raw rollout JSONL files are
preserved under `attempt_20260730/`.

## Qualification outcome

| Frozen check | Result | Gate |
| --- | ---: | ---: |
| Completed cells | 4/4 | pass |
| Raw trajectories preserved | 128/128 | pass |
| Strict four-call trajectories | 0/128 | fail |
| Parser/executable rate | 0.000 | at least 0.950 |
| Reward support | only `-1.0` | non-degenerate |
| AF/LF emitted masks distinct | no | required |
| OOM or non-finite metric | none | pass |

All 32 logged rewards were `-1.0`; all logged losses and gradient norms were
`0.0`. Every step reported 512 trainable tokens and zero masked response
tokens in both arms. The paired raw samples were also identical within each
seed. This is not evidence that AF and LF are equivalent: because every sample
failed before a valid multi-turn tool trajectory existed, the intended
supervision contrast was never instantiated.

## Scientific boundary

Q1 was an instrument and variance pilot, not an efficacy test. Its strict
executability, reward-support, and mask-distinction gates all failed. No
accuracy, stability, or sample-efficiency claim is permitted, and no
main-conference upgrade is supported.

The v1.1 protocol is consumed and terminal. Prompt changes, parser relaxation,
model changes, sampling changes, repaired calls, or selective reruns would
constitute a new instrument and must not be spliced into this result. Further
work requires a newly justified high-level protocol rather than another
unfrozen retry.
