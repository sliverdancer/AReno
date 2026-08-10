# Tau3 parseability canary execution runbook

Status: `CPU_ONLY_RUNBOOK_AWAITING_SEPARATE_SINGLE_REQUEST_AUTHORIZATION`

This runbook is the handoff for the missing external public-environment
parseable tool-call canary. It does not authorize model/API/GPU execution by
itself.

## Scope

- Environment candidate: Tau3/Tau2 airline public task.
- Execution unit: exactly `1 task x 1 model x 1 rollout`.
- Retry budget: `0`.
- Success gate: parseable tool-call emission only.
- Not evaluated here: strict task success, reward-resolution calibration,
  topology comparison, training, BFCL, held-out/sealed tasks.
- Raw model response text must not be committed; only derived observation fields
  and hashes are admissible.

## Pre-authorization dry-run

Use this path before any external spending:

```powershell
python research\reward_identifiability_supervision_topology\negative_result\external_audit\tau3_airline_parseability_canary\bind_runtime_receipt.py `
  --source-commit <40_hex_commit> `
  --model-repo-or-api-id <model_or_api_model_id> `
  --model-revision <model_revision_or_api_revision> `
  --user-simulator-model <fixed_user_simulator_model> `
  --user-simulator-revision <fixed_user_simulator_revision> `
  --user-simulator-authorization-sha256 <64_hex_authorization_hash> `
  --gpu-uuid-or-api-provider <gpu_uuid_or_api_provider> `
  --task-id <tau3_airline_public_task_id> `
  --output <run_dir>\TAU3_RUNTIME_RECEIPT_BOUND.json
```

Then:

```powershell
python research\reward_identifiability_supervision_topology\negative_result\external_audit\tau3_airline_parseability_canary\run_tau3_single_request_canary.py `
  --runtime-receipt <run_dir>\TAU3_RUNTIME_RECEIPT_BOUND.json `
  --output-dir <run_dir> `
  --dry-run
```

Expected dry-run output:

- `<run_dir>\TAU3_RUNTIME_RECEIPT_BOUND.json`
- `<run_dir>\TAU3_RUNTIME_RECEIPT_BOUND.json.sha256`
- `<run_dir>\TAU3_REQUEST_PLAN_DRY_RUN.json`

No model request is sent by this path.

## Real single-request path

This path requires separate explicit authorization before use:

```powershell
python research\reward_identifiability_supervision_topology\negative_result\external_audit\tau3_airline_parseability_canary\run_tau3_single_request_canary.py `
  --runtime-receipt <run_dir>\TAU3_RUNTIME_RECEIPT_BOUND.json `
  --output-dir <run_dir> `
  --execute-one-request `
  --base-url <openai_compatible_base_url> `
  --api-key <api_key>
```

Expected terminal outputs:

- `<run_dir>\TAU3_CANARY_OBSERVATION.json`
- `<run_dir>\TAU3_CANARY_TERMINAL_FINALIZER.json`

Terminal interpretations:

- `PASS_PARSEABLE_TOOL_CALL`: a parseable tool call was emitted. This opens only
  the next protocol-design step for reward-resolution calibration.
- `TERMINAL_PARSE_FAILURE`: no parseable tool call was emitted. This closes the
  Tau3 canary route unless a separately frozen non-outcome-conditioned format
  investigation is approved.

## Hard stops

Stop before the first request if any of these are true:

- Any runtime receipt value remains `UNBOUND`.
- More than one model request is needed.
- Retry would be needed.
- Raw response text would need to be committed.
- The task is not a public Tau3/Tau2 airline task.
- BFCL, held-out/sealed data, or training would be touched.
- The endpoint cannot expose OpenAI-style tool calls.

