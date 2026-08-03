# RIST T0b v1.2 execution report

Status: `PASS_TWO_FAMILY_RUNTIME_TREATMENT_QUALIFICATION`

The exact `3c28f06` archive, fresh v1.2 tasks, compiled CUDA extension, and all
locked model files passed fail-closed verification before serving. No model was
downloaded or replaced.

Qwen3-0.6B completed all 32 fresh rows across eight four-turn tasks in 52.82
client seconds. Gemma4 E2B completed all 32 rows in 156.51 client seconds. Both
clients used zero retry. Their response-token journals are independently bound
by SHA256 in `FINAL_RESULT.json`.

After both servers stopped, CPU-only production fixture capture used fresh
isolated tokenizer snapshots and only the corresponding v1.2 journal. Qwen and
Gemma each passed all 32 localization and exact name-only cases. No v1.0 or
v1.1 task or response entered this scientific result.

Qwen served for 151 seconds and Gemma for 280 seconds, totaling 431 of the
authorized 1,800 seconds. Both servers stopped and the final GPU process list
was empty. No training, held-out access, BFCL content access, model replacement,
or scientific request retry occurred.

This is a positive measurement qualification, not evidence that AF/LF/AN/LN
changes task success. It opens C0 mixed-reward resolution and E1 optimizer-step
capacity only after separate authorization. The main-conference route remains
unupgraded until stable seed-level, token-matched, cross-family, cross-algorithm,
and Tau3 outcomes exist.
