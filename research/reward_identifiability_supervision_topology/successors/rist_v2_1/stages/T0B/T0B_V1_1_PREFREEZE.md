# T0b runtime-token protocol v1.1 prefreeze

Status: `CPU_PREFREEZE_AWAITING_PUBLIC_API_FIX_AUTHORIZATION`

T0b v1.1 is a new protocol, not a repair or retry of terminal v1.0. It may be
frozen only after every CPU regression in
`PUBLIC_SERVE_RESPONSE_METADATA_CHANGE_REQUEST.md` passes.

The new runtime protocol must retain the two immutable model revisions, native
attention, eager decode, thinking disabled, one running prompt, temperature
zero, and zero scientific retries. It must generate eight fresh calibration
tasks with new nonces and a new task hash; no v1.0 task or response may enter the
v1.1 fixture. Each model must independently contribute exactly 32 valid rows,
eight per each of four turns, with actual non-empty engine response token IDs.

Qualification requires both models to pass the production exact name-only mask
on all 32 rows, with zero tool-name boundary mixed tokens. Any HTTP error, OOM,
missing metadata, ambiguous multi-choice metadata, malformed tool call, or
incomplete turn coverage terminates the new model cell without retry. A pass
only opens C0/E1; it is not evidence of a supervision-topology effect and does
not upgrade the project to a main-conference route.

No GPU execution is authorized by this prefreeze. A later execution manifest
must separately freeze the new serving source commit, task hash, model hashes,
time ceiling, and no-training/no-held-out/no-BFCL boundaries before requesting
GPU authorization.
