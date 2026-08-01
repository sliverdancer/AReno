# P3 auditor freeze

Freeze state: `FROZEN_BEFORE_HELDOUT_SOURCE_OPEN`

Freeze date: `2026-08-01`

Held-out framework: OpenRLHF commit
`bc71bb19464aca306b33080b2d2bb45d154e2f49`.

At this point the held-out repository had not been cloned or inspected. The
following files are immutable for the one-shot held-out evaluation:

| File | SHA-256 |
|---|---|
| `research/silent_reward_contracts/arca.py` | `f1805dc3104fcab82cd5d22d984379c9ea3f9c322ab17de07d9103d14657952c` |
| `research/silent_reward_contracts/evaluate_auditor.py` | `6dab53958717c01d9eba27762245be6aad6730661d7f7f345a8e5a784c575152` |
| `research/silent_reward_contracts/p3/dev_cases.json` | `c90c7ced9dbe45fdd9f50471dfb68b4a133ec18cba69eadbd954b286a6935e8d` |

## Frozen score

- development cases: `163`;
- macro recall: `1.0`;
- clean false-positive rate: `0.0`;
- high-severity natural recall: `1.0`;
- natural conclusion flips detected: `2`;
- wall-time limit: `60` seconds, passed.

The development JSON/CSV were regenerated twice and compared byte-for-byte.
No rule, threshold, exception, case schema, or scoring code may change after
opening held-out source. Only a source adapter and held-out cases may be added.
Any required change returns `INVALID_P3_HELDOUT_CONTAMINATION`.
