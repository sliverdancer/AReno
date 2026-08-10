# RIST negative-result handoff

Status: `NEURIPS_ED_RESEARCH_PACKAGE_CPU_ONLY_BFCL_RECEIPT_FROZEN`

This directory reframes the terminal RIST v3.1/v4.0 evidence as a negative
result about reward-resolution collapse under group-relative objectives. It is
not a v4 continuation plan and does not authorize any new GPU serving,
qualification, held-out/BFCL access, or training.

Read order:

1. `NEGATIVE_RESULT_POSITIONING.md`
2. `FORMALISM.md`
3. `RELATED_WORK_THREAT_MATRIX.md`
4. `COLLAPSE_TAXONOMY.md`
5. `DIAGNOSTIC_CHECKLIST.md`
6. `PAPER_OUTLINE.md`
7. `EXTERNAL_AUDIT_PROTOCOL.md`
8. `external_audit/bfcl_v3_base_multiturn/FEASIBILITY_AUDIT_REPORT.md`
9. `external_audit/bfcl_v3_base_multiturn/PUBLIC_FILE_MANIFEST.json`
10. `external_audit/bfcl_v3_base_multiturn/SCHEMA_SUMMARY.json`
11. `external_audit/bfcl_v3_base_multiturn/EXECUTION_RECEIPT_STATIC.json`
12. `external_audit/bfcl_v3_base_multiturn/PRE_INFERENCE_RISK_ASSESSMENT.md`
13. `submission/NEURIPS_ED_DRAFT.md`
14. `EVIDENCE_SUMMARY.csv`
15. `EVIDENCE_SUMMARY.json`

The key empirical fact is stable across two independent task-pool lineages:

- `RIST-C0-v3.1`: 2,048 calibration trajectories, zero common high-resolution cells.
- `RIST-C0-v4.0`: 2,048 calibration trajectories, zero common high-resolution cells.

The current target is a NeurIPS Evaluations & Datasets style evaluation
methodology paper. The package includes the formalism, related-work threat
matrix, diagnostic checklist, draft submission text, reproducible figures, a
frozen external-audit protocol, a CPU-only BFCL public-split feasibility audit,
and a static execution receipt. Do not retune v4 after observing the calibration
outcomes.

The next admissible step is CPU-only BFCL evaluator wiring and synthetic replay.
Any model/API/GPU inference still requires separate authorization. Raw BFCL
prompt, possible-answer, and function-documentation files are not committed in
this repository; only hashes, schema summaries, selected public task ids, and
the static execution receipt are tracked.
