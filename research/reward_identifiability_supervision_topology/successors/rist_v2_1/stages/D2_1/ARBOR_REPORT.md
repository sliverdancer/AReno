# RIST-v2.1 D2.1 Arbor report

Initial material: parent D2 at `49b09ec`, archived score 7/8.

Final best: H1, branch `arbor/rist-v2-1-h1`, executor commit `fdb26a5`,
merged into the successor branch as `62cce39`.

| Node | Hypothesis | Dev | Test | Result |
|---|---|---:|---:|---|
| n1 | exact split bytes are the reproducibility truth | 8/8 | 8/8 | merged |
| n2 | normalize JSON-domain fragments plus exact bytes | 8/8 | unopened | explored |

The tie was resolved before qualification using the predeclared simplicity
rule. Qualification was consumed once in a fresh detached worktree. No test
result informed another candidate or edit.

Reusable insight: split-local exact bytes are a transferable reproducibility
contract; language-runtime container types are serialization metadata, not
scientific evidence.
