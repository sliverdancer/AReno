# SAS-TR-v2.1 Successor Plan — Serving Capacity Repair

Status: `CPU_PLAN_FROZEN_GPU_AUTHORIZATION_REQUIRED`

Parent: terminal invalid `SAS-TR-v2.0` B2 attempt

## Registered defect

`SAS-TR-v2.0` started AReno serving with `max_running_prompts=8`. On the 24GB
RTX 4090D this allocated about 22.5 GiB for model state and KV cache. The first
request failed while allocating another 642 MiB. No model completion was
observed, so the attempt contains no scientific evidence about tool readiness.

## Allowed repair

The successor changes serving capacity and failure handling only:

1. set AReno `--max-running-prompts 1`;
2. set the client to one concurrent trajectory;
3. run one disposable health-plus-single-request preflight before consuming
   any frozen cell row;
4. require that preflight to return a complete OpenAI response object without
   CUDA OOM, worker exit, or HTTP 500;
5. fail the controller immediately if any cell file has a request error or
   incomplete raw evidence;
6. handle SIGINT/SIGTERM by marking the controller aborted and terminating the
   serving process group.

No checkpoint, model weight, tokenizer thinking condition, response budget,
tool schema, task split, seed, sampling policy, eligibility gate, selection
order, or reward rule may change. All four calibration cells must restart from
zero; no v2.0 output may be spliced into v2.1.

## GPU gate

This is a new protocol version. It remains unopened until the repair has CPU
regression coverage, a fresh source archive and manifest are frozen, and the
user explicitly authorizes a new bounded GPU serving attempt.
