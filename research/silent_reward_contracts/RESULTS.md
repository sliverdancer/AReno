# Frozen CPU results

## Natural evidence

| Framework | Natural case | Severity | Ordinary execution | ARCA | Conclusion flip |
|---|---|---|---|---|---|
| AReno | string arguments vs dict-only reward | high | successful, reward 0 | F1 | yes |
| veRL | `tool_rewards` producer vs `reward_scores` consumer | high | successful, reward contribution absent | F2 | yes |
| veRL | timeout mapped to numeric 0 without status | medium | successful return | F6 | no |
| AReaL | missing reward mapped to numeric 0 without status | medium | successful tensorization | F6 | no |

The issue #199 `last_assistant` and `final_answer` masks were identical on the
registered well-formed trajectory by design. They are retained as a declared
equivalence control, not counted as a natural failure.

## Frozen evaluator

| Split | Cases | Natural failures | Mutation failures | Clean controls | Macro recall | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|
| Development | 163 | 3 | 80 | 80 | 1.00 | 0.00 |
| Held-out OpenRLHF | 80 | 0 | 40 | 40 | 1.00 | 0.00 |
| Post-freeze AReaL | 1 | 1 | 0 | 0 | exact match | n/a |

The held-out hierarchical bootstrap interval is `[1.00, 1.00]`, which is
mechanically inevitable under perfect detection in the registered cells and
does not express uncertainty over unseen frameworks. Wilson intervals in the
machine-readable metrics should be used for per-rule finite-sample reporting.

## Baselines on held-out mutations

| Baseline | Macro recall | Clean FPR |
|---|---:|---:|
| Exit code | 0.00 | 0.00 |
| Finite-number check | 0.00 | 0.00 |
| Schema only | 0.00 | 0.00 |
| OpenRLHF native preflight | 0.25 | 0.00 |
| ARCA | 1.00 | 0.00 |

## P4 dynamic validation

P4-v0.2 attempted its first frozen command and stopped after 12.82 seconds
during reward-module import. No model was loaded, no trajectory was generated,
and no update ran. The result is
`INVALID_P4_INFRASTRUCTURE_REWARD_IMPORT`, not a scientific failure. Five runs
remain unopened and v0.2 may not be resumed.

No P4 model execution, serving, paid API, external issue, or pull request was
performed. The measured failed-command occupancy was `0.00356` single-GPU
hours.
