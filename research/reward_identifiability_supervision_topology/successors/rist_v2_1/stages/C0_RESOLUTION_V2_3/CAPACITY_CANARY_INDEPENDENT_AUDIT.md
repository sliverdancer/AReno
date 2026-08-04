# RIST C0 v2.3 capacity-canary independent CPU audit

Decision: `PASS_CPU_INSTRUMENT_ONLY_MODEL_REQUESTS_NOT_AUTHORIZED`

The final read-only audit inspected HEAD
`1b83201d397f7723b5c7f2a051388e8b8a779007` and the frozen source commit
`7b2bffd61c1a96ed73ed45137cf2c204015e51ed`.

The audit independently confirmed:

- the post-rent binding protocol and both deployment-receipt protocols;
- the exact six-field receipt identity schema;
- receipt authority equality with the post-rent binding authority;
- commit ancestry and all eight current and committed artifact hashes;
- atomic fresh-journal reservation and the exact eight frozen request seeds;
- live GPU UUID and memory checks before requests; and
- a strict single-process predicate, with the sole live compute process equal
  to the Python executable bound in the family receipt.

The exact CPU command covering the capacity harness, deployment replay, and
fresh-pool construction completed with `26 passed`. The commit-backed capacity
freeze verifier reported `passed=true`, `boundary_pass=true`,
`binding_ancestor_pass=true`, and `binding_semantics_pass=true`, with all eight
current and committed files passing.

Adversarial tests confirmed that a different external GPU process and two
processes using the same expected Python path both fail before journal
reservation, thread-pool creation, or any `post_json` call.

Boundary: this audit approves only the CPU instrument and its freeze. It does
not authorize GPU/model requests, serving, scientific split access,
held-out/BFCL access, or training.
