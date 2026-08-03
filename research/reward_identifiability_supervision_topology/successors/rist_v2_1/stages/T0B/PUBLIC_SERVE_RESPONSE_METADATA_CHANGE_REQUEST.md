# Public serving response-metadata change request

Status: `CPU_DESIGN_FROZEN_AWAITING_EXPLICIT_PUBLIC_API_AUTHORIZATION`

## Problem

T0b v1.0 consumed one request for each frozen model and failed because the HTTP
response lacked actual `areno.response_tokens`. Code inspection identifies two
independent omissions at runtime commit `0f09715`:

1. `_build_response_from` calls `build_chat_completion_response` without
   `include_areno_metadata=True` or `input_tokens=prompt`, so the shared builder
   never creates the extension in serve mode.
2. `ChatCompletionResponse` has no `areno` field, so Pydantic drops an injected
   extension during response serialization.

Fixing only either omission is insufficient. The consumed v1.0 protocol must
not be repaired or rerun.

## Minimum additive public change

After explicit authorization, add a typed optional response extension:

```python
class ArenoResponseMetadata(BaseModel):
    input_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]

class ChatCompletionResponse(BaseModel):
    # Existing fields remain unchanged.
    areno: ArenoResponseMetadata | None = None
```

For a single-choice (`n == 1`) serve response, `_build_response_from` must call
the existing shared builder with `include_areno_metadata=True` and
`input_tokens=prompt`. `response_tokens` must be the unmodified engine token IDs
for the sole choice. `response_logprobs` must remain empty because serve does
not compute them; values must never be reconstructed from decoded text.

For `n > 1`, the extension must remain absent until a separately designed
per-choice schema exists. This avoids silently labeling the first choice's token
IDs as metadata for all choices. No request field, CLI option, sampling rule,
tool parser, or OpenAI-compatible field may change.

## Required CPU regression tests

1. A single-choice `_build_response_from` preserves exact prompt and response
   token IDs through `model_dump`/FastAPI serialization.
2. The extension is absent for `n > 1` rather than ambiguous.
3. `response_logprobs` is exactly empty and is never fabricated.
4. Existing tool-call parsing output is byte-equivalent apart from the additive
   `areno` field.
5. A cancelled/empty response exposes an empty token list, which the T0b client
   still rejects.
6. The OpenAPI response schema marks `areno` optional, preserving backward
   compatibility.

## Authorization boundary

This document does not authorize editing `areno/cli/serve.py`, changing the
public HTTP response, running inference, accessing models, training, opening
held-out/BFCL content, or using a GPU. The next action requires explicit public
API authorization under `AGENTS.md`.
