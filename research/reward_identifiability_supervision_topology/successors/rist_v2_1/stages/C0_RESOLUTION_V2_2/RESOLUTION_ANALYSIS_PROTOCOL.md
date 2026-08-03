# RIST C0 v2.2 Resolution Analysis Protocol

This CPU-only stage begins only after the content-blind C0 v2.2 collection
finalizer returns the exact decision
`PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS`. It consumes the already
collected 4,096 trajectories; it does not serve a model, train, access held-out
data, or access BFCL.

## Irreversible access boundary

The runner first recomputes the content-blind collection final, verifies the
fresh-pool and D3-train bindings, and requires a new output directory. It then
creates `OUTCOME_ACCESS_RECEIPT.json` with exclusive-create semantics before it
interprets actions or rewards. Every collection result, trajectory journal, raw
response journal, source manifest, D3 train artifact, and evaluator is bound by
SHA-256. A post-receipt error writes a terminal KILL result; the same directory
must never be repaired or rerun.

## Outcome replay

For every task/seed pair, the runner parses each raw assistant message with the
frozen D4 strict-call evaluator. It stops at the first invalid call and requires
the submitted trajectory fields to match the replay with exact value and type.
Each family/split must cover exactly 32 tasks by 32 unique rollout seeds.

## Frozen resolution rule

Seeds are sorted and divided into four groups of eight per task. A group is
mixed when it contains both strict rewards. Each structural cell contains four
tasks and therefore 16 groups. The cell is:

- collapsed when the 95% Wilson upper bound is at most 0.25;
- resolved when the 95% Wilson lower bound is at least 0.50;
- transition otherwise.

A collapsed/resolved label also needs at least three of four tasks to agree;
otherwise it becomes heterogeneous. A cell transports only when calibration and
qualification have the same collapsed/resolved label. Each family needs at
least two transported low cells and two transported high cells. The common map
keeps only same-band cells shared by both families and again needs at least two
cells per band.

## E1 admission

On PASS, D3 train data are filtered by whole structural cells only. The E1
capacity dataset is all four tasks from the lexicographically first common high
cell. `E1_ADMISSION.json` binds the collection, common map, D3 source, filtered
dataset, four E1 tasks, model revisions, tokenizer snapshots, and GPU UUID. It
is not executable authorization. The independent validator must replay the
entire evidence chain and exclusively create `VALIDATION_RESULT.json`; the E1
gate then reproduces that validation again. Only the exact validation decision
`PASS_C0_V2_2_TO_E1_CAPACITY` can proceed to a separate deployment-bound E1
manifest. Collection PASS, resolution PASS, or an admission file alone cannot
open serving or training.
