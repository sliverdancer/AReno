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

## Runtime templates

`TAU3_RUNTIME_RECEIPT_TEMPLATE.json` is deliberately unbound and non-executable.
It records the exact fields that must be bound before the first request and
requires pre-request exit if any runtime value remains `UNBOUND`.

`TAU3_OBSERVATION_SCHEMA.json` records the only admissible single-request
observation shape. It forbids raw response text in committed artifacts,
additional requests, nonzero retry, reward-resolution claims, BFCL access,
held-out/sealed access, and training.

## Runtime binder

`bind_runtime_receipt.py` converts the unbound runtime template into a bound
single-request receipt. It validates the frozen template hash, rejects any
remaining `UNBOUND` value, enforces 40-hex source commits and 64-hex
authorization hashes, and writes a SHA-256 sidecar. It is still CPU-only and
does not start Tau3 or send a model request.

## Single-request runner

`run_tau3_single_request_canary.py` is the execution harness for the future
authorized canary. By default it writes only `TAU3_REQUEST_PLAN_DRY_RUN.json`.
The real path requires `--execute-one-request`, a valid bound runtime receipt,
an OpenAI-compatible `base_url` and API key, and still enforces one request and
zero retry. It writes a derived observation plus terminal finalizer; raw model
response text remains out of tracked artifacts.
If no OpenAI-compatible serving stack is available, the runner also supports
`--local-transformers-model-path`, which performs exactly one local
`transformers.generate` call against a fixed cached model snapshot.

## Dry-run evidence

`build_dry_run_evidence.py` generates `dry_run_evidence/` with a synthetic bound
receipt, its SHA-256 sidecar, a dry-run request plan, and a manifest. This
artifact proves the pre-authorization execution chain is wired without sending a
model request. It is not a model result and cannot satisfy the parseable
external canary by itself.
