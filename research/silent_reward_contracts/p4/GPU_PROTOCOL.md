# P4 frozen dynamic contract validation

Protocol: `ARCA-P4-DYNAMIC-v0.1`

State: `AWAITING_NEW_GPU_INSTANCE_AND_EXPLICIT_AUTHORIZATION`

P4 asks one narrow question: does canonicalizing the exact reward boundary
restore an informative reward in a live agentic rollout while the historical
strict contract remains silently zero? It does not test CARe and cannot support
a model-capability or sample-efficiency claim.

## Frozen design

- one verified local Qwen3-0.6B snapshot;
- one A800-80GB or equivalent GPU;
- identical 64-example dataset, agent, optimizer settings, and one update;
- two arms: `strict` and `canonical` reward boundary;
- paired seeds `3101`, `3102`, `3103`;
- only `--reward-fn-path` and output directory differ within a pair;
- sequential run order: strict then canonical within each seed;
- maximum 60 minutes per run, 6 GPU hours, 8 instance hours, CNY 60;
- stop after the first crash, timeout, non-finite metric, dirty checkout, source
  hash mismatch, or ceiling violation.

## Primary outcome and gate

The primary outcome is reward-contract informativeness at step 0/1, supported
by `reward_mean`, `reward_std`, `trainable_tokens`, and
`masked_response_tokens`. Return `PASS_P4_DYNAMIC_CONTRACT_CONSEQUENCE` only if
all six runs and artifacts are complete, every strict run has exactly zero
reward mean and standard deviation, and at least one canonical run has nonzero
reward mean or standard deviation. Otherwise return
`KILL_P4_NO_DYNAMIC_CONTRACT_CONSEQUENCE`; do not repair or selectively rerun.

Any learning comparison, longer training, new model, or new framework is a new
protocol. The historical CARe P3-v0.2 result remains terminal and cannot be
spliced into P4.
