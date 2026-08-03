# T0b runtime-token protocol v1.1 prefreeze

Status: `COMPLETED_AS_SEPARATE_T0B_V1_1_CPU_FREEZE`

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

The completed freeze is in sibling stage `T0B_V1_1`. Its manifest freezes the
new serving source commit, task hash, client hashes, model lock, proposed time
ceiling, and no-training/no-held-out/no-BFCL boundaries. It retains
`execution_authorized=false`; GPU execution still requires separate user
authorization.
