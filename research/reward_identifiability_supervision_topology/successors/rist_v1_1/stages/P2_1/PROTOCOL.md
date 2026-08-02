# P2.1 Cross-Family Reward-Resolution Qualification

Protocol: `RIST-P2.1-v1.0`

Frozen: pending CPU freeze

Status: `DRAFT_CPU_FREEZE`

## Purpose

Test whether the analytic reward-resolution range frozen in P1 is realized by
two actual model families before any supervision-topology training is opened.
P2.1 is inference-only. It may qualify or kill the task instrument, but it
cannot estimate AF/LF/AN/LN efficacy.

This is a successor to terminal `RIST-P2-v1.0`, which produced zero scientific
responses and ended `INVALID_P2_INFRASTRUCTURE`. It does not repair or splice
that run. `RIST-E0-v1.2` subsequently passed the independent native-backend
interface canary for both checkpoints; P2.1 uses that exact serving stack.

## Frozen prerequisites

Before any qualification row is loaded by a serving client:

1. the archived E0-v1.2 stage result and independent audit hashes must match
   the execution manifest and both must report PASS;
2. the registered `areno.accel._areno_accel` extension, CUDA, editable source,
   and `areno check` must pass as in E0-v1.2;
3. both already-downloaded checkpoint file manifests must pass in full;
4. the GPU process listing must be empty;
5. both servers must use `--attn-backend native --eager-decode
   --disable-thinking --max-running-prompts 1`;
6. no qualification request may be used as a health check or environment
   repair. Each model must return `{"status":"ok"}` on `/health` first.

Any failure before a scientific response returns
`INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`. Do not retry, change backend,
replace a checkpoint, or consume a qualification row for debugging.

## Frozen checkpoints and order

Run sequentially on one RTX 4090D:

1. `Qwen/Qwen3-0.6B` (`qwen3_0_6b`);
2. `google/gemma-4-E2B-it` (`gemma4_e2b_it`).

Use only the snapshots whose complete file manifests were revalidated under
E0-v1.2. Downloads, substitutions, tokenizer changes, adapter changes, and
checkpoint writes are forbidden.

## Frozen data and sampling

- data: the 32-row P1 qualification split only;
- held-out: forbidden and unopened;
- rollout seeds: `6101, 6202, 6303, 6404, 6505, 6606, 6707, 6808`;
- trajectories: 256 per checkpoint, 512 total;
- action opportunities: exactly four per trajectory;
- temperature: `1.0`; top-p: `1.0`; thinking: disabled;
- maximum new tokens: 128 per action;
- requests are sequential and `max-running-prompts=1`;
- zero retries after a scientific request reaches the server;
- zero call repair, synthesis, or parser fallback;
- all four raw responses are retained even after strict success becomes
  impossible.

The request-seed derivation, prompt construction, forced/free tool exposure,
deterministic failure observations, strict reward, and analysis gates are
byte-preserved from the unconsumed scientific client in RIST-P2-v1.0 except
for protocol validation and additional integrity accounting.

## Primary qualification measures

Per checkpoint and task:

- first-turn executable-call rate;
- four-turn parsed-call completion rate;
- strict exact-oracle success;
- unique full action sequences across eight seeds;
- mixed strict-reward group indicator;
- non-zero group-relative-advantage trajectory count;
- empirical reward entropy and advantage-collapse rate;
- raw/parsed agreement and invalid-reason counts.

## Frozen gates

Each checkpoint must satisfy:

1. first-turn executable rate at least `0.90`;
2. four-turn parsed-call completion rate at least `0.75`;
3. overall strict success in `[0.05, 0.95]`;
4. at least `6/32` mixed strict-reward task groups;
5. at least `48/256` non-zero-advantage trajectories;
6. zero fabricated/repaired calls or retries and complete raw evidence.

The cross-family instrument must additionally satisfy:

1. both checkpoints pass every interface and signal gate;
2. at least two analytic strata contain at least two mixed groups per
   checkpoint;
3. at least one task per checkpoint has multiple action sequences but a
   homogeneous strict reward;
4. qualification outputs are not used to modify any frozen choice.

## Outcomes and stopping

- `PASS_P2_1_CROSS_FAMILY_RESOLUTION_TO_P3_PROTOCOL_FREEZE`;
- `KILL_P2_1_TASK_INSTRUMENT_NO_MODEL_RESOLUTION`;
- `KILL_P2_1_CHECKPOINT_INTERFACE`;
- `INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`.

Any KILL closes the current main-conference route. Any invalid outcome closes
this protocol revision without a scientific claim. A changed task, seed,
checkpoint, backend, parser, threshold, request order, or resource ceiling
requires a new protocol revision.

## Resource forecast and authorization boundary

E0-v1.2 measured 20.964 seconds for eight Qwen requests and 81.528 seconds for
eight Gemma requests. Scaling the frozen sequential client to 1,024 requests
per checkpoint forecasts approximately 3.65 GPU-hours before safety margin.
The prior two-hour estimate is therefore rejected as infeasible before opening
scientific data.

P2.1 requests a hard ceiling of five GPU-hours for sequential serving. CPU
preparation may continue, but server launch and qualification access require a
new explicit authorization for this protocol and ceiling. No optimizer,
training, checkpoint save, qualification-driven repair, or held-out access is
permitted.
