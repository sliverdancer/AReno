# RIST negative-result handoff

Status: `NEURIPS_ED_RESEARCH_PACKAGE_WITH_PARSEABLE_POSITIVE_CONTROL`

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
17. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_LOCAL_PREFLIGHT_BLOCKED.md`
18. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_LOCAL_PREFLIGHT_BLOCKED.json`
19. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_TERMINAL_REPORT.md`
20. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_TERMINAL_FINALIZER.json`
21. `external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_RUNTIME_RECEIPT_BOUND.json`
22. `external_audit/bfcl_v3_base_multiturn/FORMAT_INVESTIGATION_REPORT.md`
23. `external_audit/bfcl_v3_base_multiturn/FORMAT_INVESTIGATION_SUMMARY.json`
24. `external_audit/bfcl_v3_base_multiturn/FORMAT_REPAIR_REPORT.md`
25. `external_audit/bfcl_v3_base_multiturn/FORMAT_REPAIR_REPLAY_RESULT.json`
26. `external_audit/bfcl_v3_base_multiturn/REPAIRED_FORMAT_CANARY_TEMPLATE.json`
27. `external_audit/bfcl_v3_base_multiturn/REPAIRED_CANARY_TERMINAL_REPORT.md`
28. `external_audit/bfcl_v3_base_multiturn/REPAIRED_CANARY_TERMINAL_FINALIZER.json`
29. `external_audit/agentic_tictactoe_positive_control/POSITIVE_CONTROL_REPORT.md`
30. `external_audit/agentic_tictactoe_positive_control/POSITIVE_CONTROL_RESULT.json`
31. `external_audit/tau3_airline_parseability_canary/TAU3_PARSEABILITY_CANARY_REPORT.md`
32. `external_audit/tau3_airline_parseability_canary/TAU3_PARSEABILITY_CANARY_TEMPLATE.json`
33. `external_audit/tau3_airline_parseability_canary/TAU3_RUNTIME_RECEIPT_TEMPLATE.json`
34. `external_audit/tau3_airline_parseability_canary/TAU3_OBSERVATION_SCHEMA.json`
35. `external_audit/tau3_airline_parseability_canary/TAU3_CANARY_FINALIZER_REPLAY.json`
36. `submission/NEURIPS_ED_DRAFT.md`
37. `EVIDENCE_SUMMARY.csv`
38. `EVIDENCE_SUMMARY.json`

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

The remote minimal canary executed exactly one Qwen3-0.6B model request and
terminated as an interpretable parse failure: zero parseable tool calls, strict
success false, zero retries. This does not open the full BFCL external audit or
the two-model canary. The follow-up format investigation identifies a probable
format-layer issue, not a reward-resolution result. Raw BFCL prompt,
possible-answer, function-documentation files, and raw model response are not
committed in this repository; only hashes, schema summaries, selected public
task ids, static execution receipt, synthetic replay results, canary receipts,
terminal summaries, format-investigation summaries, and CPU-only format-repair
fixtures are tracked. The next admissible inference step is a newly authorized
one-request repaired-format canary; that canary has now also terminated as a
parse failure. Full BFCL audit and two-model canary remain closed.

The package now also includes a separate CPU-only positive control using the
public repo-native `examples/agentic/tictactoe` `choose_square` tool protocol.
It produces parseable synthetic tool calls and confirms that the diagnostic can
classify all-pass, all-fail, and mixed reward groups when the tool-call format
is valid. This is not a BFCL result, not a model result, and not a substitute
for a true external public benchmark canary; it only closes the narrower concern
that the analyzer itself might be unable to pass on parseable tool-call data.

The next frozen external candidate is Tau3/Tau2 airline. Its CPU-only canary
template binds the existing `examples/agentic/rist_v2_1_tau3` adapter and
validates, with synthetic OpenAI-style tool-call fixtures, that the adapter can
convert exactly one parseable tool call into Tau3 action JSON. The template
remains unbound and does not authorize model/API/GPU/training execution.
Its terminal finalizer is also frozen and replay-tested for both parseable PASS
and terminal parse-failure outcomes. Runtime receipt and observation templates
are present but deliberately unbound and non-executable until separate
single-request authorization is given.
