# BFCL minimal canary prompt/tool-call format investigation

Status: `FORMAT_LAYER_ISSUE_PROBABLE_FULL_AUDIT_CLOSED`

Scope:

- CPU-only/local static investigation.
- No new model request.
- No API use.
- No GPU use.
- No training.
- No held-out/sealed access.
- Raw model response not read or committed.

## Evidence used

- `MINIMAL_CANARY_TERMINAL_FINALIZER.json`
- `MINIMAL_CANARY_RUNTIME_RECEIPT_BOUND.json`
- `MINIMAL_CANARY_REMOTE_HASH_MANIFEST.json`
- Public BFCL schema summaries and file hashes.
- The repository-side parser and one-request canary runner logic.

## Terminal fact

The minimal Qwen3-0.6B canary sent exactly one model request and terminated as:

- `failure_mode = PARSE_FAILURE`
- `observed_call_count = 0`
- `parseable_tool_calls = false`
- `strict_success = false`
- `retry_count = 0`
- `go_to_two_model_canary = false`

This is an interface/format failure before reward-resolution evidence can be
estimated.

## Most likely format risks

1. BFCL function docs use `parameters.type = "dict"`, whereas most
   OpenAI-style and Transformers chat-template tool renderers expect JSON Schema
   `type = "object"`. Passing BFCL docs through unchanged may produce a prompt
   that is visible but not behaviorally aligned with the model's learned tool
   format.
2. The canary used direct `transformers.generate` with rendered tools but no
   serving-layer tool-choice enforcement. Therefore the model was free to answer
   in prose even when the prompt requested tool calls.
3. The parser accepts Qwen-style `<tool_call>{...}</tool_call>` and raw JSON
   tool-call objects. Because zero calls were parsed, the failure is not a
   strict-success miss; it is a tool-call emission or parser-interface failure.
4. The first BFCL task turn expects three tool calls. That is valid BFCL
   behavior but stricter than a single-call format canary. It may be too hard as
   the first interface canary even if the model can emit one valid tool call.

## What cannot be concluded

- This does not show Qwen3-0.6B cannot solve BFCL tasks.
- This does not show reward-resolution collapse on BFCL.
- This does not justify running more rollouts.
- This does not justify opening the full 64-task audit.

## Blocked tokenizer probe

The next diagnostic would normally render the exact Qwen tokenizer
chat-template prompt and record structural facts only. The remote SSH endpoint
closed the connection during tokenizer-only probing, so this step is not yet
available in the artifact.

## Recommendation

Do not run more rollout sampling. The next admissible work is CPU-only format
repair design:

1. Convert BFCL `parameters.type = "dict"` to JSON Schema `type = "object"` in
   a derived prompt adapter.
2. Add a synthetic, no-model parser fixture for the exact expected Qwen tool
   block.
3. Freeze a new one-request format canary using a single expected tool call,
   not a three-call first turn.
4. Only after that canary emits parseable tool calls should a two-model canary
   be considered.
