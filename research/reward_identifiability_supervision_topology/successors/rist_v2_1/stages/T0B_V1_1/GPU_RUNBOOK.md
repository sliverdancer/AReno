# T0b v1.1 GPU runbook

Status: `CPU_PREPARED_NOT_EXECUTION_AUTHORIZED`

This runbook is descriptive and does not change
`EXECUTION_MANIFEST.json:execution_authorized=false`. After a separate user GPU
authorization, execution must deploy the exact `2cca319` Git archive verified by
`verify_deployment_preflight.py`. It must reuse only already locked model
snapshots after the existing snapshot verifier passes; downloads and revision
replacement remain forbidden.

The GPU must be empty before Qwen starts. Verify the tokenizer hashes, start one
native/eager/thinking-disabled server with one running prompt, wait for health,
and invoke the v1.1 client exactly once. Journal each raw response before
validation. Stop the server and verify no GPU process before starting Gemma.
Apply the same sequence to Gemma. Total serving time may not exceed the separately
authorized ceiling and may never exceed the frozen requested maximum of 1,800
seconds.

Any missing or empty `areno.response_tokens`, HTTP failure, malformed call, OOM,
incomplete 32-row coverage, source/hash mismatch, or timer violation terminates
the affected cell with no retry. Do not patch the server, regenerate tasks,
reuse v1.0 inputs, or change model/backend/sampling settings during execution.

After both cells stop, copy all journals, results, server logs, timestamps,
snapshot verifications, and GPU-empty checks back to a new evidence directory.
Only then build production mask fixtures and run the T0 qualification hook on
CPU. No result may open C0/E1 unless both models independently pass all 32 rows.
