# P2 Cross-Family Inference Qualification Report

Protocol: `RIST-P2-v1.0`

Terminal decision: `INVALID_P2_INFRASTRUCTURE`

Scientific interpretation: `NOT_ESTIMABLE`

## Outcome

P2 did not produce a model trajectory and therefore does not test reward
resolution, action diversity, the AF/LF/AN/LN interaction, or the research
hypothesis. It is neither a scientific pass nor a scientific kill.

The Qwen3 server passed its HTTP health check but returned HTTP 500 before the
first scientific response on two consecutive attempts:

1. the frozen source was initially missing its compiled `areno_accel` runtime
   extension;
2. after the documented editable CUDA build succeeded and `areno check`
   reported the extension ready, the default attention path failed because
   `flash_attn` was not installed.

Both failures occurred at task index 0, seed 6101, before any assistant output
was returned. The second failure triggered the protocol's frozen
`REPEATED_SERVER_FAILURE` stop rule. Switching to the native attention backend,
installing another dependency, or restarting qualification after that point
would have been an outcome-contingent repair, so none was attempted.

## Integrity and resource accounting

- execution source: `9176964a0394be1b5f54c7166a8ca033cd0b4318`;
- protocol/manifest commit: `c70aaae`;
- qualification SHA-256:
  `8018137606e12da0f0096ac86f11312d94d631326965198783ebf9cecc94570f`;
- Qwen snapshot-file manifest SHA-256:
  `ffe4fab0fed7e695aff4f17b876a091e0f984674da045d96e681f835c94715f0`;
- Gemma snapshot-file manifest SHA-256:
  `74c6a28bb8465bc710cef8fb88e9d921f7bc1114aec363d12acb99c5d2f65b7f`;
- conservative GPU window: 379 seconds of the authorized 7,200 seconds;
- final GPU state: 3 MiB allocated, 0% utilization;
- trajectories: Qwen 0, Gemma 0;
- scientific responses: 0;
- fabricated or repaired calls: 0;
- training: not performed.

The complete 50 KiB evidence archive has SHA-256
`e5cf647466963a8fb7015a499b2c6271e448f9d37dc93aa0d8ac646c7b5c388c`.
Its extracted files pass the included per-file SHA-256 manifest.

## Main-conference hook

The stage is invalid, so the hook returns `INVALID_PROTOCOL_STOP`. The route
cannot be upgraded to a main-conference candidate from this evidence. A future
attempt requires a newly frozen protocol that preflights both the compiled
extension and the chosen attention backend before consuming the qualification
run. This report does not authorize that attempt or any training.
