# RIST C0 v2.3 independent CPU audit

Decision: `PASS_CPU_FREEZE_SAFE_TO_RENT_GPU_NOT_AUTHORIZED_TO_SERVE`

The read-only audit used freeze commit
`5a9a62c6dceffd80a35916137b0f68c390b3bcd4`, source commit
`defce3614fbea3763a690cb40c34931059b4e54b`, and runtime commit
`f00b688dff3e4ad35229d059536c975d4c56594f`.

The audit independently confirmed fresh and reproducible task and rollout
seeds, one deployment entrypoint, a pre-rent authority over control/runtime
commits, manifest SHA-256, and model revisions, and a post-rent receipt design
that binds GPU UUID and extension SHA-256. All six identities are remeasured
before the append-only ledger records `launch_attempt`. Runtime launch receives
embedded manifest bytes and no manifest path.

Evidence: 16 v2.3 CPU tests passed, including two coordinator-held merge gates;
the commit-backed freeze verifier passed all 11 current and committed files. An
additional deletion replay confirmed that launch does not reopen the source
manifest. Wrong worktrees, individual and combined identity mismatches,
receipt tampering, and recomputed internal self-hashes produced zero launcher
calls.

The audited raw SHA-256 values are:

- `CPU_FREEZE.json`: `ba58b0ed9ab9d6704d29d9089bfee3a65e4bb75e683a3e4e6e8b17b21cc59487`
- `PRE_RENT_AUTHORITY.json`: `6541cbdf723e8ea3bc13315d1b90452303ebd67eb344c470abef2d51b1a36b54`
- authority canonical SHA-256: `8790a2f3092b936a176b93ae4882e2e6568b66a215e8b76f4db9f4d843caf19f`

Residual boundary: GPU rental may begin, but serving remains unauthorized.
After rental, the exact receipt must bind the observed GPU UUID and extension
SHA-256, and its canonical on-disk artifact SHA-256 must be externally frozen
before the same entrypoint may launch anything.

The auditor also observed one pre-existing failure in the broader legacy RIST
suite concerning a `rist_v2/D2` generator-version expectation. The involved
files are unchanged from the v2.3 baseline; it is neither a v2.3 blocker nor
v2.3 scientific evidence.
