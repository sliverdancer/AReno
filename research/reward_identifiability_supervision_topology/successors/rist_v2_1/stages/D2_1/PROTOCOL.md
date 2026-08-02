# RIST-v2.1 D2.1 split-local evaluator freeze

Protocol: `RIST-D2.1-v2.1`

Decision: `PASS_D2_1_CPU_POOL_TO_DOWNLOAD_AUDIT`

The frozen successor evaluator reads exactly one split and regenerates exactly
that split. It does not accept a data directory or manifest, and it never calls
the parent three-split generation entrypoints.

Arbor used calibration for development and consumed qualification once for the
merge gate. Both predeclared candidates scored 8/8 on development. The frozen
tie rule selected H1, which treats exact JSONL bytes as the reproducibility
truth and does not let Python container representation metadata veto identical
bytes. H1 then scored 8/8 on qualification.

The selected candidate hash is
`3ddc7fcbabc4eb7d28284dde47e4a6fb3f7267705f384fa3e3918a12f963c302`.
The qualification input hash is
`07a56fc3662ff1e1260bdbf3fe32b2312f6c2a577848c48d5c47c1c000fbe82a`.

No held-out content, model, GPU, serving, inference, training, checkpoint, or
download was accessed. PASS validates the CPU structural evaluator only.
