# T0b actual runtime response-token gate

Status: `FROZEN_AUTHORIZED_AWAITING_REACHABLE_REMOTE`

T0b is an inference-only treatment-identifiability canary. It runs Qwen3-0.6B
then Gemma4 E2B sequentially with the T0a tokenizer revisions and the archived
native serving stack. It uses eight fresh, explicit calibration nonces, four
turns per nonce, and retains exactly 32 raw OpenAI-compatible responses per
model. These tasks are not D3 training, qualification, held-out, Tau3, or BFCL
content and have no task-success interpretation.

Before each server starts, the serving checkpoint's allowlisted tokenizer files
must match the T0a per-file hashes. Model download, replacement, tokenizer
substitution, parser repair, request retry, training, and checkpoint writes are
forbidden. The GPU must be idle before launch. Each server uses native attention,
eager decode, thinking disabled, and one running prompt. The server is stopped
and GPU process absence verified before the next model.

Any missing actual `areno.response_tokens`, malformed exact tool call, HTTP
failure, OOM, tokenizer mismatch, or incomplete 8-per-turn coverage stops that
model without retry. T0 passes only if both frozen runtime fixtures independently
report 32 cases, eight per turn, exact name-only masks, zero name-boundary mixed
tokens, and `qualification_pass=true`.

The total serving ceiling is 1,800 GPU-seconds. This authorization does not open
C0, E1 optimizer canaries, P3 training, held-out evaluation, or BFCL content.
