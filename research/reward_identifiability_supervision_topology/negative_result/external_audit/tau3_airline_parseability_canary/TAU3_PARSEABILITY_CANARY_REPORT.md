# Tau3 airline parseability canary freeze

Status: `CPU_FROZEN_AWAITING_SEPARATE_SINGLE_REQUEST_AUTHORIZATION`

This is the next external public-environment candidate after the BFCL
format route terminated at parse failure. It is CPU-only and freezes a
future `1 task × 1 model × 1 rollout` canary. No model/API/GPU/training
action is authorized by this file.

## CPU parser replay

- Replay status: PASS
- Adapter: `examples/agentic/rist_v2_1_tau3/run_agent.py::response_to_action`
- Parsed tool name: `DB`
- Malformed non-object arguments rejected: True

## Runtime gate

A future execution must bind the source commit, model/API identity, Tau3
user-simulator identity, runtime target, and exactly one public airline task
before the first request. The only success criterion is parseable tool-call
emission; strict reward success and reward-resolution claims remain closed.

## Terminal finalizer

`tau3_canary_finalizer.py` validates a bound runtime receipt and finalizes a
single supplied observation. The CPU-only replay at
`TAU3_CANARY_FINALIZER_REPLAY.json` covers both terminal outcomes:

- `PASS_PARSEABLE_TOOL_CALL` when at least one observed call has a string name
  and object arguments;
- `TERMINAL_PARSE_FAILURE` when the single request emits no parseable tool call.

The finalizer rejects training authorization, BFCL use, held-out/sealed access,
nonzero retry, task/model mismatch, and more than one model request.
