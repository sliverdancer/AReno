# E0 Independent Native-Backend Serving Canary

Protocol: `RIST-E0-v1.1`

Frozen: `2026-08-01`

Status: `FROZEN_AWAITING_GPU_AUTHORIZATION`

## Purpose

Validate the AReno serving stack before any qualification set is exposed. E0
is an infrastructure and structured-output canary only. It cannot estimate
reward resolution, supervision-topology efficacy, or any scientific outcome.

The canary is required because `RIST-P2-v1.0` terminated before producing a
single scientific response: the first server attempt lacked the compiled
`areno_accel` extension and the second used the default flash backend without
`flash_attn`. E0 isolates those failures from the successor qualification set.

## Strict separation from scientific data

- use only `canary_tasks.json`;
- do not read P1 train, qualification, or held-out rows during GPU execution;
- instructions disclose the exact expected tool and exact code, so the result
  is not a capability evaluation;
- E0 outputs may decide whether the serving stack is usable, but may not be
  pooled with RIST-P1/P2/P3 evidence;
- the terminal `RIST-P2-v1.0` record remains immutable and may not be repaired
  or spliced into this protocol.

## Frozen checkpoints and order

Run sequentially on one RTX 4090D:

1. `Qwen/Qwen3-0.6B` (`qwen3_0_6b`);
2. `google/gemma-4-E2B-it` (`gemma4_e2b_it`).

Use the already downloaded, hash-recorded snapshots from the terminal P2
evidence. Do not download or substitute another checkpoint under E0.

## Frozen serving configuration

For both checkpoints:

- `--tp-size 1 --world-size 1`;
- `--attn-backend native`;
- `--eager-decode`;
- `--disable-thinking`;
- `--max-running-prompts 1`;
- one model resident at a time;
- no fallback to the flash backend;
- no retry after a canary request reaches the server.

Before launching either model, the frozen-source environment must pass:

1. `python -c "import areno_accel"`;
2. `python -c "import torch; assert torch.cuda.is_available()"`;
3. `areno check` with the imported editable source matching the execution
   manifest commit;
4. a clean GPU process listing with no unrelated serving or training process.

After launch, require `/health` to return `{"status":"ok"}` before any canary
request. A preflight or health failure terminates E0 without a model request.

## Canary workload

Each model receives two independent four-turn trajectories:

- one forced-function trajectory with one offered function per turn;
- one required/free-function trajectory with the instructed function plus one
  distractor per turn.

Each turn explicitly states the expected function and code. Sampling is
deterministic (`temperature=0`, `top_p=1`, frozen request seed), with at most 96
new tokens. The client performs no tool-call repair, synthesis, parser fallback,
or retry. Every raw response is retained.

## Frozen gates

Each checkpoint must produce:

1. 8/8 raw chat-completion responses;
2. 8/8 exactly one parseable offered tool call;
3. 8/8 exact instructed tool-and-code pairs;
4. zero HTTP 5xx or client infrastructure exceptions;
5. zero fabricated or repaired calls.

Both checkpoints must pass. The validator returns exactly one of:

- `PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE`;
- `FAIL_E0_MODEL_INTERFACE_STOP`;
- `INVALID_E0_PREFLIGHT_STOP`;
- `INVALID_E0_INFRASTRUCTURE_STOP`.

Any non-PASS outcome closes this E0 revision. It does not support a scientific
claim. A changed backend, checkpoint, source commit, task byte, or threshold
requires a new protocol revision.

## Resource and authorization boundary

CPU preparation and tests are authorized in the current stage. Actual preflight
on the remote GPU host, serving, and model requests require separate approval.
The requested GPU ceiling for E0 is 30 minutes total. Training, optimizer
creation, checkpoint writes, scientific qualification data, and held-out access
are forbidden.
