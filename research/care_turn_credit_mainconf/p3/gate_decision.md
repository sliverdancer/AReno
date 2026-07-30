# P3 CPU-freeze gate decision

Protocol: `CARE-P3-PILOT-v0.1`
Decision date: `2026-07-30`
Decision: `PASS_P3_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`
GPU status: `BLOCKED_PENDING_EXPLICIT_USER_AUTHORIZATION`

## What passed

1. A non-reference CARe router now uses disjoint calibration and update
   trajectory blocks from one frozen behavior-policy batch.
2. Calibration rows receive zero update budget; update rows use a fixed
   256-token denominator with no backfill.
3. The `care` and `uncalibrated` arms use the same cheap scorer, exact audits,
   batch shape, model, seeds, and optimizer settings. Only the calibrated
   threshold application differs.
4. Invalid calls are retained, never synthesized, and pessimistically handled.
5. Exactly two arms by three qualification seeds are generated.
6. The collector aligns per-step reward, `trainable_tokens`,
   `masked_response_tokens`, routed mass, audit calls, split counts, validity,
   and threshold into CSV and JSON.
7. ModelScope asset drift is fail-closed through 11 frozen file sizes and
   SHA-256 hashes because the hub exposes only a mutable `master` branch.
8. The executor rejects dirty/mismatched source, remote checkpoint references,
   changed asset manifests, model hash drift, missing runs, and execution
   without the explicit command-line flag.
9. Hard resource rules are frozen: AutoDL A800 80GB, one hour per run, six
   training GPU-hours, eight billed instance-hours, and CNY 60 total.

## CPU evidence

Observed commands and results:

```text
python -m pytest tests/test_care_p3_design_cpu.py -q
10 passed

python -m pytest \
  tests/test_care_p1_cpu.py \
  tests/test_care_turn_credit_cpu.py \
  tests/test_care_p3_design_cpu.py \
  tests/test_trainable_turns_ablation_cpu.py \
  tests/test_metrics_cpu.py \
  tests/test_train_cli_config_cpu.py \
  tests/test_agentic_cpu.py \
  tests/test_trainer_api_cpu.py -q
190 passed

git diff --check
passed
```

A real CPU preparation was generated twice at the same path and compared
byte-for-byte:

- dataset SHA-256:
  `bdb6fd0f502a9f7a7bb2e2947a09c89d884ffada7364fb92c81d4105c38fa03d`;
- CPU-freeze manifest SHA-256:
  `85dd558660502980f50444ceaba532392657981058637ad873f1b74f5ebeb97d`.

The manifest hash is not a remote-run identifier. A final manifest must be
regenerated from the clean reviewed commit and verified checkpoint path.

## Scientific interpretation

This pass establishes design and artifact executability on CPU fixtures only.
It does not establish:

- GPU memory fit or runtime;
- successful model tool calls;
- non-degenerate model reward;
- a finite optimizer step;
- post-update learning;
- real-task transfer;
- or method efficacy.

The experimental-design audit forced the calibration/update split, whole-block
units, matched audit costs, invalid-row retention, and explicit external-
validity limits. These controls prevent the P3 pilot from being presented as
stronger evidence than it is.

## Remaining blocker before provisioning

The reviewed source is packaged on the dedicated
`research/trainable-turns-ablation` branch. The immutable source identifier is
the commit containing this document; record it with `git rev-parse HEAD` after
checkout. The GPU host must use a clean detached checkout of that exact
revision. Local unrelated untracked work is not part of the research artifact.

Before GPU work:

1. fetch the dedicated branch and record its exact commit;
2. verify the detached remote checkout is clean;
3. recheck the live AutoDL A800 rate, assigned CPU/RAM, and storage charge;
4. regenerate the manifest outside the checkout with the verified local model
   path;
5. run the read-only remote preflight;
6. obtain explicit user approval for the GPU command.

## Next legal action

After an AutoDL instance is rented, run only the inventory and preparation
steps in `AUTODL_HANDOFF.md`. Report the clean commit, verified model hashes,
live quote, and dry-run preflight. Then ask the owner whether to authorize the
six-command pilot on one AutoDL A800 80GB under the CNY 60 / eight-hour cap.
Remote preparation authorization must not be interpreted as GPU-training
authorization.
