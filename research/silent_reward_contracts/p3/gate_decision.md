# P3 gate decision

Decision: `PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST`

Date: `2026-08-01`

All preregistered conditions passed: frozen hashes matched, every registered
high-severity natural development case was detected, held-out mutation macro
recall was `1.00`, clean held-out false-positive rate was `0.00`, two natural
development cases changed a scientific conclusion or gate, and CPU workloads
completed under 60 seconds.

The evidential boundary is strict: OpenRLHF contributed 40 source-derived clean
cases and 40 synthetic single-fault cases, but no natural failure. AReaL added
one post-freeze natural replication and does not alter the held-out estimate.
P4 is not authorized by this pass; it waits for a new GPU instance, an explicit
budget ceiling, and a separate execute instruction.
