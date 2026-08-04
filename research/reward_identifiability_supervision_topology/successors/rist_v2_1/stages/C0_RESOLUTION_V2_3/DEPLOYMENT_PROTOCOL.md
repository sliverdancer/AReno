# RIST C0 v2.3 deployment authority

Before renting a GPU, freeze one authority template and its external SHA-256.
The template fixes the only allowed control commit, runtime commit, manifest
SHA-256, and model revisions. `create-receipt` must consume that exact template
and refuse mismatched worktrees or data before producing an artifact.

After renting, the same entrypoint may discover the concrete GPU UUID and
extension SHA-256 into the receipt. The exact receipt artifact SHA-256 must then
be frozen and explicitly authorized before `launch`. Launch re-probes all six
identities and records its decision in the append-only ledger. Any authority,
receipt, or live-identity mismatch exits without a `launch_attempt`; serving is
structurally downstream of acceptance.

The runtime never accepts or resolves a manifest path. This CPU replay does not
load a model, contact a network service, start serving, or use a GPU.
