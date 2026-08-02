# RTX 4090D release handoff

Date: `2026-08-02`

Before release, the remote GPU and relevant-process listings were empty and the
remote evidence archive SHA-256 matched the local copy:
`8c958d65be56948055b1b22a0fac043c7bbec2b612d48481203fdd8f094ee098`.

The subsequent SSH sync attempt and a fresh read-only SSH probe were both
closed by the endpoint before a session was established. This is consistent
with the instance being stopped or detached. There is no remaining authorized
RIST GPU work.

Codex does not have AutoDL account/API access, so platform billing or instance
allocation cannot be independently confirmed from SSH. The user should verify
the AutoDL console shows the instance as shut down/released. Local scientific
evidence is complete and does not depend on the remote disk.
