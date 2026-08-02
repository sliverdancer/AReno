# Tau3 upstream test retirement

During authorized development metadata inspection, upstream
`get_tasks("train")` loaded the unified domain `tasks.json` before filtering by
the IDs in `split_tasks.json`. Consequently, bytes belonging to upstream test
tasks entered the Python process even though no test task content was printed,
selected, evaluated, or used to tune an artifact.

Fail-closed decision: the Tau3 upstream `test` split is permanently retired as
a confirmatory source for RIST. Tau3 may provide environment qualification,
development evaluation, and real-environment training only. Sealed external
confirmation remains exclusively BFCL, whose `bfcl_eval/data` directory was not
materialized during acquisition or installation.
