# Artifact contract

Every machine-readable result must contain a schema/protocol identifier, source
or manifest provenance, execution state, and deterministic case/run identity.

| Artifact | Required identity | Required measurements |
|---|---|---|
| P1 production cases | case, source path | observed/oracle value, detection |
| P2 cross-system cases | framework, class, source hash | observed/oracle, severity, conclusion flip |
| P3 case evaluation | framework, workload, seed, natural flag | expected/predicted rules, exact match |
| P3 gate summary | frozen hashes, split | recall, FPR, intervals, baselines, evidence class |
| P4 step result | arm, seed, step, manifest hash | reward mean/std, trainable and masked tokens |

CSV is the flat analysis view; JSON is authoritative for nested provenance.
All numeric values must be finite. Generators reject missing tags, duplicate
steps, unaligned arms, source-hash drift, and frozen-evaluator drift. Raw logs
must be archived alongside, not embedded into, the summary artifacts.
