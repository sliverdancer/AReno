# RIST negative-result handoff

Status: `NEURIPS_ED_RESEARCH_PACKAGE_CPU_ONLY_MINIMAL_CANARY_TEMPLATE_FROZEN`

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
13. `external_audit/bfcl_v3_base_multiturn/SYNTHETIC_REPLAY_REPORT.md`
14. `external_audit/bfcl_v3_base_multiturn/SYNTHETIC_REPLAY_RESULT.json`
15. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json`
16. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_RUNTIME_RECEIPT_REPORT.md`
17. `submission/NEURIPS_ED_DRAFT.md`
18. `EVIDENCE_SUMMARY.csv`
19. `EVIDENCE_SUMMARY.json`

The key empirical fact is stable across two independent task-pool lineages:

- `RIST-C0-v3.1`: 2,048 calibration trajectories, zero common high-resolution cells.
- `RIST-C0-v4.0`: 2,048 calibration trajectories, zero common high-resolution cells.

The current target is a NeurIPS Evaluations & Datasets style evaluation
methodology paper. The package includes the formalism, related-work threat
matrix, diagnostic checklist, draft submission text, reproducible figures, a
frozen external-audit protocol, a CPU-only BFCL public-split feasibility audit,
a static execution receipt, a CPU-only synthetic evaluator replay, and a minimal
one-task one-model canary runtime receipt template. Do not retune v4 after
observing the calibration outcomes.

The next admissible step, if separately authorized, is binding the minimal
canary template to a concrete runtime environment and running exactly one model
request. Full BFCL external audit remains closed. Raw BFCL prompt,
possible-answer, and function-documentation files are not committed in this
repository; only hashes, schema summaries, selected public task ids, static
execution receipt, synthetic replay results, and canary receipt template are
tracked.
