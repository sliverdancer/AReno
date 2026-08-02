# E1 per-checkpoint capacity and runtime gate

Status: `FROZEN_AWAITING_MODEL_AND_GPU_AUTHORIZATION`

Capacity is qualified separately for each checkpoint and GPU type. A serving
health response is insufficient for training authorization.

## Required order

1. Verify exact model revision and tokenizer hash without replacing either.
2. Native-backend serving canary on calibration-only prompts: 32 tasks, four
   turns, raw responses journaled before validation, zero retry.
3. One optimizer-step canary on D3 train data for GSPO and GRPO separately,
   using the largest arm by trainable-token mass.
4. Verify positive trainable tokens, finite loss/gradient norm, no OOM, at least
   15 percent memory headroom, and checkpoint save/load roundtrip.
5. Destroy canary outputs after hashing only if the future execution protocol
   explicitly requires it; otherwise retain immutable evidence.

`validate_capacity_evidence.py` admits a checkpoint only when all serving and
training requirements pass. Inference success cannot substitute for an
optimizer-step canary.

## Current evidence boundary

The archived RIST-v1.1 run supports Qwen3-0.6B serving on a 24 GB 4090D. It does
not qualify training. Gemma4 E2B returned HTTP 500/CUDA OOM on its first
trajectory and therefore rejects that checkpoint/GPU pairing. This is an
infrastructure result, not model-quality evidence.

The current two-family matrix therefore cannot execute on that 24 GB pairing.
There are only two admissible routes:

1. retain Gemma4 E2B and pass a fresh E1 canary on a different, larger-memory
   GPU; the protocol does not infer a sufficient memory size from the OOM; or
2. open a new versioned design and qualify a smaller non-Qwen checkpoint from
   tokenizer snapshot through actual runtime tokens, parser behavior, serving,
   and both one-step training canaries.

AReno's Llama adapter and generic JSON parser establish implementation
availability only. With no frozen checkpoint-specific tokenizer/template or
runtime fixture, they are not evidence that a Llama-family substitute is
scientifically qualified. A silent checkpoint substitution is forbidden.
