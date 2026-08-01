# P3 — frozen transfer evaluation

Decision: `PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST`

The ARCA rules, evaluator, and development cases were hashed before OpenRLHF
source was opened. The one-shot held-out set contains 40 source-derived clean
controls and 40 registered single-fault mutations across reward flow, grouped
advantages, treatment identity, and failure status.

| Result | Value |
|---|---:|
| Held-out macro recall | 1.00 |
| Clean false-positive rate | 0.00 |
| OpenRLHF native-preflight mutation recall | 0.25 |
| Exit/finite/schema-only mutation recall | 0.00 |
| CPU wall-time gate | pass |

OpenRLHF exposed no natural failure under the registered source inspection.
Therefore the 1.00 held-out recall is mutation-transfer evidence, not a claim
of natural OpenRLHF vulnerability prevalence. After the one-shot evaluation,
an optional AReaL production-method probe independently detected one natural
medium-severity failure-status conflation. It is reported as replication and
is not included in the held-out score.

Reproduce with `REMOTE_CPU_COMMANDS.md`. Machine-readable evidence is under
`artifacts/`; source pins and hashes are in `upstream_manifest.json`.
