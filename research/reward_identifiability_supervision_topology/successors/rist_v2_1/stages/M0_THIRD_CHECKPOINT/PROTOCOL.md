# M0 outcome-blind third-checkpoint protocol

Status: `PRIMARY_CANDIDATE_PREREGISTERED_NOT_QUALIFIED`

## Purpose

The main-conference route requires at least three checkpoints across at least
two model families. Qwen3-0.6B and Gemma4 E2B supply only two checkpoints. M0
pre-registers the third-checkpoint choice before any C0 v2.2, E1, P3, or Tau3
outcome is observed, preventing performance-based model shopping.

This CPU-only stage does not download a config, tokenizer, or weight; does not
serve, infer, or train; and does not add a dependency. Repository adapter
availability is implementation feasibility only, not checkpoint qualification.

## Outcome-blind selection rule

Candidates must, in order:

1. use an adapter and checkpoint loader already registered in the frozen AReno
   source and require no dependency outside the existing `pyproject.toml`;
2. be a public text-only dense causal LM rather than multimodal, MoE, or hybrid
   architecture, so a new architecture confound is not introduced;
3. preserve the already satisfied two-family design while adding within-family
   scale replication, which makes a second Qwen3 checkpoint preferable to an
   unqualified third family;
4. be the smallest published Qwen3 dense checkpoint strictly larger than the
   existing 0.6B checkpoint, limiting compute while materially separating scale;
5. have no selection input derived from benchmark score, tool-call success,
   C0 reward resolution, E1 capacity, gradient, or training outcome.

Under this rule, the provisional primary is `Qwen/Qwen3-1.7B`. The sole
pre-scientific fallback is `Qwen/Qwen3-4B`, and it may be considered only if the
primary fails metadata/license/availability checks before any model inference or
scientific outcome exists. After tokenizer/runtime/C0/E1 access begins, failure
does not permit checkpoint substitution within this paper; it closes the
three-checkpoint main-track gate. Capacity failure may be retried only by the
prospectively frozen larger-GPU escalation for the same checkpoint.

## Qualification gates

Every gate is separately authorized and fail-closed:

1. **M0 (this stage): CPU registry audit.** Verify the existing Qwen3 dense
   adapter, checkpoint I/O, and dependency manifest by source hash. This does
   not qualify any external checkpoint.
2. **M1: tokenizer/config-only metadata.** With new authorization, download only
   a pinned revision's allowlisted config/tokenizer/license files. Require
   `model_type=qwen3`, dense text architecture, compatible chat template, exact
   tokenizer hash, license compatibility, and no remote-code/new-dependency
   requirement. Freeze the exact revision before weights.
3. **M2: weight identity.** With separate authorization, download only the M1
   revision, hash every weight shard, and verify adapter load mapping without
   replacement. No inference or training is implied.
4. **M3: runtime treatment qualification.** Run the same fresh four-turn and
   production-mask qualification used by T0b v1.2, with zero retry. Both
   `full` and `name_only`, and all/last action-span treatments, must be distinct.
5. **M4: C0 resolution transport.** Run all frozen C0 v2.2 calibration and
   qualification cells for this checkpoint. Every cell used for replication
   must transport its band and agree with the preregistered cross-family map;
   no task or rollout is selected individually.
6. **M5: E1 capacity.** On a prospectively bound GPU, complete genuine one-step
   GSPO and GRPO canaries with positive gradient, checkpoint roundtrip, raw
   evidence hashes, and at least 15% memory headroom.
7. **M6: replicated outcomes.** Only after the original two-checkpoint P3/P4
   gate remains scientifically open, extend AF/LF/AN/LN to this checkpoint with
   the powered paired-seed count and the same step- and token-matched analyses.

The third checkpoint counts toward the main-conference matrix only after M1-M6
all pass. A CPU adapter match, tokenizer fixture, serving canary, or one-step
capacity result alone does not count.
