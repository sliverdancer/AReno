# Q1 Bounded Pilot Preparation

Status: `PREPARED_GPU_UNAUTHORIZED`

The bounded integration pilot is frozen as four qualification-only runs:

- supervision arms: `AF` (`all_assistant`) and `LF` (`last_assistant`);
- independent base seeds: `1101` and `2202`;
- maximum training steps: `8` per run;
- model: `Qwen/Qwen3-0.6B` from ModelScope;
- algorithm: GSPO;
- held-out outcomes consumed: **no**.

The exact commands and qualification-data hash are recorded in
`pilot_manifest.json`. Every command has an explicit `--seed`; the only
intended command differences are the seed, supervision arm, and isolated metric
directory.

Preparation verification on 2026-07-29:

- 18 structured-action/seed CPU tests passed;
- 389 repository CPU tests passed with the optional serving test module
  excluded;
- syntax compilation and `git diff --check` passed;
- the current WSL session reported `torch.cuda.is_available() == False` and
  zero CUDA devices;
- NVML initialization was blocked by the operating system.

No model download, training, serving, or GPU execution occurred. Q1 remains
open and must not receive a `stage_result.json` or main-track assessment until
all four frozen runs and their prespecified pilot analyses complete. Even a
successful Q1 can only remain diagnostic; the main-track hook forbids promotion
before a valid Q3 confirmatory result.
