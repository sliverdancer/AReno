# E1 C0 v2.2 admission gate

The previously frozen E1 v2.1 runbook and execution templates are not directly
executable with C0 v2.2 evidence. Before constructing any deployment-bound E1
manifest, run `validate_c0_admission.py` against the terminal C0 v2.2 resolution
directory. It must independently reproduce the full C0 validation and return
`PASS_C0_GATE_TO_DEPLOYMENT_BOUND_E1`.

A C0 collection final, resolution final, `E1_ADMISSION.json`, or capacity
dataset alone is insufficient. The saved `VALIDATION_RESULT.json`, C0 final,
admission, and capacity dataset hashes must all match. The gate output remains
`training_execution_authorized=false`: a separate manifest must bind the actual
GPU UUID, memory, model paths, frozen revisions, and existing user training
authorization before serving or a single optimizer step can run.
