# RIST-v2 D2 terminal report

Formal decision: `KILL_RIST_V2_D2_STRUCTURAL_POOL`

The CPU candidate generator produced 32 calibration, 32 qualification, and 32
sealed held-out tasks across eight balanced structural cells. Seven of eight
frozen gates passed. The `byte_identical_regeneration` gate returned false, so
the current protocol revision is terminal.

Read-only forensic reproduction found:

- generated manifest bytes were identical;
- calibration, qualification, and held-out JSONL bytes were identical;
- the in-memory manifest comparison was false because freshly generated
  `cell_specs` is a tuple while JSON reload converts it to a list.

The defect is in the frozen evaluator's representation comparison. It does not
show that the task pool is scientifically invalid, and it provides no evidence
for or against the reward-resolution moderator or supervision-topology
interaction. Nevertheless, the registered D2 decision is not changed,
repaired, or selectively rerun.

No GPU, model, serving, inference, training, checkpoint download/replacement,
or held-out selection occurred. A future continuation must be a new protocol
revision with the evaluator corrected and independently tested before its gate
is consumed.
