# Tau3 single-request canary authorization packet

Status: `AWAITING_USER_AUTHORIZATION_NOT_EXECUTABLE`

This packet exists to prevent scope drift when the remaining external
parseability canary is authorized. It is not an authorization by itself.

## Exact authorization text to request

Authorize execution of the Tau3/Tau2 airline parseability canary for exactly
`1 public airline task x 1 model x 1 rollout`, zero retry, using the current
research branch commit and the frozen artifacts under
`external_audit/tau3_airline_parseability_canary/`.

Permitted actions:

- install only dependencies required to run the frozen single-request canary;
- bind `TAU3_RUNTIME_RECEIPT_BOUND.json` from the frozen template;
- run `run_tau3_single_request_canary.py --execute-one-request` once;
- write only derived artifacts:
  - `TAU3_RUNTIME_RECEIPT_BOUND.json`;
  - `TAU3_RUNTIME_RECEIPT_BOUND.json.sha256`;
  - `TAU3_CANARY_OBSERVATION.json`;
  - `TAU3_CANARY_TERMINAL_FINALIZER.json`;
  - hash manifests or concise terminal reports.

Forbidden actions:

- more than one model request;
- any retry;
- training;
- BFCL access;
- held-out or sealed task access;
- reward-resolution calibration;
- topology comparison;
- committing raw model response text;
- modifying the protocol after the first request.

Success criterion:

- `TAU3_CANARY_TERMINAL_FINALIZER.json.status == "PASS_PARSEABLE_TOOL_CALL"`.

Failure criterion:

- `TAU3_CANARY_TERMINAL_FINALIZER.json.status == "TERMINAL_PARSE_FAILURE"`, or
  any pre-request hard stop.

## Recommended first model

Use the cheapest already-available OpenAI-compatible model endpoint that
supports tool calls. If using local GPU serving, prefer Qwen3-0.6B only if the
serving stack exposes OpenAI-style `tool_calls`; otherwise use an API endpoint
with native tool-call support for this one-request canary.

## Budget

- Requests: exactly 1.
- Retries: 0.
- GPU time, if local serving is used: cap at 15 minutes.
- API spend, if API is used: cap at the minimum practical budget for one chat
  completion request.

## Pre-request checklist

- `git rev-parse HEAD` is recorded.
- Runtime receipt has no `UNBOUND` values.
- Runtime receipt validates with `tau3_canary_finalizer.py`.
- `TAU3_REQUEST_PLAN_DRY_RUN.json` has been generated.
- API endpoint supports OpenAI-style tool calls.
- Raw response text will not be committed.

## Post-request checklist

- Exactly one request was sent.
- Retry count is zero.
- Observation matches `TAU3_OBSERVATION_SCHEMA.json`.
- Terminal finalizer exists.
- Result is reported as either PASS parseable tool call or terminal parse
  failure.
- No reward-resolution or task-success claim is made from this canary.

