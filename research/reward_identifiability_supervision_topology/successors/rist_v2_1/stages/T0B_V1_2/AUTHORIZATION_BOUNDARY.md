# T0b v1.2 authorization boundary

Status: `CPU_FROZEN_NOT_GPU_AUTHORIZED`

T0b v1.2 is a fresh two-model runtime-token and production-mask qualification.
It must not use any v1.0 or v1.1 task or response as scientific input. The
preserved v1.1 Gemma journal was used only after termination to falsify and
repair the CPU instrument; it cannot be scored or reinterpreted as v1.2.

Before any model server starts, the deployed source hashes, task hash, locked
model revisions, and compiled `areno_accel` extension import and binary hash must
all pass. A missing extension terminates preflight before a scientific request.
Qwen and Gemma then run sequentially with zero retry. Each model must produce 32
balanced response-token rows and independently pass the exact production
name-only mask fixture. Failure of either cell prevents C0/E1 from opening.

A separate user authorization is required for GPU serving. Unless that grant
also says otherwise, training, checkpoint download or replacement, held-out
access, BFCL content access, task regeneration, and scientific retries remain
forbidden. The requested combined serving ceiling is 1,800 seconds.
