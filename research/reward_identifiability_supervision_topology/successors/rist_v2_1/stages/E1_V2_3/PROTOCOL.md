# RIST E1 v2.3 C0 admission protocol

Status: `CPU_FROZEN_C0_OUTCOME_BLOCKED`

This successor does not repair the terminal C0 v2.2 E1 gate. It accepts only
the one-shot `PASS_C0_V2_3_TO_E1_CAPACITY` result produced by the frozen v2.3
qualification analyzer. Calibration-only evidence, a partial family result, a
transition/heterogeneous cell, or fewer than two transported cells per band
cannot open E1.

The admission retains whole D3 structural cells from the frozen common map and
selects the lexicographically first transported high cell for the one-step AF
capacity canary. It never selects an individual task or rollout by outcome.
Exactly four tasks enter the capacity dataset.

Admission binds the qualification result, D3 manifest and bytes, filtered
training bytes, capacity bytes, exact Qwen/Gemma revisions and tokenizer
snapshots, the C0 GPU UUID, and the source commit. It authorizes no serving or
training by itself. It also emits canonical copies of the qualification
transport result and the capacity-data manifest required for independent E1
evidence recomputation.

`build_deployment_manifest.py` is the only E1 deployment builder. It consumes
that exact admission plus both family runtime identities and binds one clean
worktree, the same live GPU with at least 79 GiB, fixed model and tokenizer
revisions, model-weight and snapshot hashes, and the extension SHA. It opens
only the standing-authorized matrix Qwen/Gemma x GSPO/GRPO, AF, one complete
optimizer step. Held-out and BFCL access remain prohibited. Standing GPU
authorization never bypasses the ordered C0 GO/KILL gate.
