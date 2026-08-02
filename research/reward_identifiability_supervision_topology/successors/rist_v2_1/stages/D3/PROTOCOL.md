# RIST-v2.1 D3 train split and strict runner contract

Status: `CPU_ONLY_FREEZE`

## Train split

D3 generates 32 train tasks: four fresh nonces for each of the eight D2
structural cells. The split name participates in nonce derivation, making its
signatures disjoint from calibration, qualification, and sealed held-out.
Every cell is retained. Future model calibration may select or reject whole
cells only; it may not select individual train tasks by outcome.

## Training interface

The research-only dataset loader accepts only the D3 `train.jsonl` path and
constructs a prompt from initial visible turn contracts. The agent runner:

1. exposes every offered tool for the current turn;
2. uses required tool calling without forcing the expected tool name;
3. accepts exactly one raw call with exact `{code: string}` arguments;
4. stops on the first wrong tool, wrong code, malformed call, or missing call;
5. reveals dependent next-turn information only after a correct action;
6. performs zero retry, repair, synthesis, or parser fallback;
7. appends the raw response before validation when a journal path is set.

The strict binary reward is one only for the exact complete oracle call
sequence. Invalid and incomplete trajectories remain zero and stay in the
denominator.

## Boundary

CPU fixtures validate contracts only. Real tokenizer AN/LN boundaries,
checkpoint capacity, rollout reward resolution, and training remain unopened.
