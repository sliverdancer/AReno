# BFCL minimal canary local preflight

Status: `BLOCKED_BEFORE_FIRST_MODEL_REQUEST`

The user authorized binding and running the minimal one-task, one-model,
one-rollout canary. The receipt requires fail-fast validation before the first
model request. Local preflight failed, so no model request was sent.

## Blockers

- No local Qwen/Gemma model snapshot is available in the Hugging Face cache.
- Installed PyTorch is CPU-only and reports `cuda_available=False`.
- `vllm` is not installed.
- `accelerate` is not installed.
- No concrete remote GPU or API runtime was provided for this canary turn.
- `MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json` still contains `UNBOUND`
  runtime fields and is not executable.

## Scope actually used

- Model request sent: no.
- API request sent: no.
- GPU model serving started: no.
- Training started: no.
- Held-out/sealed access: no.
- Raw BFCL data committed: no.

## Decision

Do not execute the canary on the current local runtime. Bind a concrete runtime
first, then rerun preflight before the first model request.

Safe next options:

1. Provide SSH for a running GPU instance and authorize deploying the current
   commit plus a fixed Qwen3-0.6B revision there.
2. Or provide an API runtime and exact model id authorized for one request.
3. Keep the full 64-task BFCL audit closed until this one-request canary is
   terminal and interpretable.
