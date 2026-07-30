# Q0 Instrument Qualification Report

Decision: `PASS_Q0_INSTRUMENT`

The initially blocked result is preserved under `attempt_01_blocked/`. After
explicit authorization, one public base seed now controls:

- parent Python, NumPy, and Torch initialization;
- spawned worker initialization;
- deterministic per-epoch dataset order;
- deterministic per-step rollout sampling;
- deterministic per-sample and per-turn agent requests.

The seed is available through `TrainerConfig.seed` and CLI `--seed`. CUDA
bitwise determinism is not claimed because hardware kernels may remain
nondeterministic; the protocol records environment identity and independent
seeds.

All other Q0 checks remain satisfied:

- four distinct factorial masks and an empty `Z0` mask;
- zero-signal batches skip optimizer mutation;
- strict no-synthesis tool-call validation;
- byte-preserved model arguments;
- 48/16/16 signature-disjoint tasks;
- exact four-call reward and frozen metric schema.

Verification:

- 17 SAS-specific CPU tests passed;
- 388 CPU tests passed with the optional serving module excluded;
- full collection remains unavailable because the current virtual environment
  lacks `fastapi`;
- syntax compilation, diff checks, and evidence hashes passed.

Q1 is now the next stage, but remains unopened until explicit GPU/model
authorization. No model, held-out outcome, or GPU execution was consumed in Q0.

