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
