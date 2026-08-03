# RIST T0b v1.0 execution report

Status: `TERMINAL_FAIL_CLOSED`

Both fixed-revision snapshots passed size and digest verification, including all
T0a tokenizer hashes. The deployed `areno/cli/serve.py` SHA256 is
`37f2b72cc431efe625ddc485845fb3770f2a016b024a350bbf7dd6b4838d28e9`,
identical to commit `0f09715`; the later deployment-only commits add the locked
T0b acquisition harness and do not change runtime serving code.

Qwen3-0.6B and Gemma4 E2B were served sequentially with native attention, eager
decode, thinking disabled, and one running prompt. Each model produced the exact
requested first call to `scan_registry` with code `548c010f76ad`. In both raw
HTTP responses, however, the top-level `areno.response_tokens` extension was
absent. Each client therefore stopped after one row with zero retries, as frozen.
Qwen served for 254 seconds and Gemma for 231 seconds, totaling 485 of the
authorized 1,800 seconds. Both servers were stopped and `nvidia-smi` reported no
compute process afterward.

The failure is localized to the common response serialization path. The lower
level `build_chat_completion_response` adds `areno.response_tokens`, while the
frozen `ChatCompletionResponse` Pydantic schema has no `areno` field. A CPU-only
probe confirmed that a payload containing `areno` serializes without it. This is
an instrumentation failure, not evidence against either model, the name-only
mask, or the supervision-topology hypothesis.

T0b v1.0 is consumed and must not be repaired or rerun. The next admissible path
is an additive public response-metadata fix with a CPU regression test, followed
by a newly frozen T0b v1.1 protocol and separate GPU authorization. No training,
held-out access, BFCL content access, model substitution, or scientific request
retry occurred.
