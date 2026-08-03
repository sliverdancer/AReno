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

The training canary is the prospectively largest supervision arm, AF
(`all_assistant` plus `full`), at exactly one step, eight samples, and one
frozen seed shared by GSPO and GRPO. From the C0-filtered train hash, the canary
dataset retains all four tasks in the lexicographically first transported
high-resolution cell; this structural rule never selects an individual task or
rollout by outcome. The step must exercise a nonzero gradient;
zero trainable tokens, zero/non-finite gradient norm, missing TensorBoard or
reward journals, more than one optimizer step, any retry, or a failed
checkpoint reload rejects the pairing. All raw, metric, checkpoint-manifest,
reload, model, tokenizer, source, GPU UUID, driver, CUDA, and PyTorch identities
are retained in the evidence object.

`validate_capacity_evidence.py` admits a checkpoint only when all serving and
training requirements pass. Inference success cannot substitute for an
optimizer-step canary.

Qualification is recomputed from the original TensorBoard event directory,
the exact eight-row reward journal, the 32 raw four-turn training responses,
the sampled GPU-monitor trace, the actual saved-checkpoint directory, and the
checkpoint-reload journal. A submitted metrics summary, peak-memory scalar,
`checkpoint_roundtrip` boolean, or other derived field is never authoritative.
Its bytes are retained, but it must equal the independently recomputed value.
The deployment manifest must replace the GPU placeholder with one concrete UUID;
that UUID, the checkpoint name and revision, tokenizer and weight hashes, source
commit, driver, CUDA, PyTorch, and raw monitor identity must agree everywhere.
Qwen requires a GPU reporting at least 24 GB-class memory and Gemma at least
48 GB-class memory; neither the manifest nor a summary may weaken these floors.

## Current evidence boundary

T0b v1.2 now supports Qwen3-0.6B and Gemma4 E2B serving plus production-mask
qualification on a 24 GB 4090D. It does not qualify training. An older Gemma4
run OOMed on that pairing; the later serving pass retires the serving-only
rejection but provides no evidence of optimizer capacity. Both checkpoint/GPU
training pairings therefore remain unqualified.

The current two-family matrix cannot execute until E1 passes. There are only
two admissible routes for Gemma:

1. retain Gemma4 E2B and pass a fresh E1 canary on a different, larger-memory
   GPU; the protocol does not infer a sufficient memory size from the OOM; or
2. open a new versioned design and qualify a smaller non-Qwen checkpoint from
   tokenizer snapshot through actual runtime tokens, parser behavior, serving,
   and both one-step training canaries.

AReno's Llama adapter and generic JSON parser establish implementation
availability only. With no frozen checkpoint-specific tokenizer/template or
runtime fixture, they are not evidence that a Llama-family substitute is
scientifically qualified. A silent checkpoint substitution is forbidden.
