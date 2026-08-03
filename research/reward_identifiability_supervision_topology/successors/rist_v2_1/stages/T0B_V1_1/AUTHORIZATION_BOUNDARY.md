# T0b v1.1 execution boundary

Status: `AWAITING_SEPARATE_GPU_AUTHORIZATION`

The public serving response extension and CPU regressions are complete. The
frozen manifest still sets `execution_authorized=false`; this stage performed no
model access, inference, training, GPU action, held-out access, or BFCL content
access.

The next admissible authorization may open only sequential serving of the two
already locked revisions, using the frozen v1.1 tasks and source commit, for at
most the requested 1,800 GPU-seconds. It must not authorize training, model
substitution, new downloads, v1.0 task reuse, held-out data, or BFCL content.
