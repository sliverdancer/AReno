# P5 external natural-validation results

Decision: `PASS_P5_EXTERNAL_NATURAL_TO_PAPER`

Five previously unseen, source-pinned agentic-RL systems were retained through
inspection. Two qualifying natural cases were reproduced by compiling and
executing the exact relevant function bodies from the pinned upstream files:

| Framework | Boundary | Frozen rule | Result | Scope |
|---|---|---|---|---|
| slime | official forge evaluation replay | F6 status/provenance | `reward=None` becomes `0.0`, `truncated=False`, with no reward-present field | medium severity; eval-replay, not a core training-path prevalence claim |
| rLLM | documented lightweight Python evaluator coercion | F3 informativeness | `is_correct=False` and `is_correct=True` dicts without `reward` both become reward `0.0`; explicit reward restores `1.0` | high severity because the documented evaluator output feeds training reward |

The rLLM behavior appears in two separate production coercion functions and is
counted once. Agent-R1, RAGEN, and Agent Lightning remain bounded negative
inspections: their audited fallbacks retained structured error/status evidence
or were explicit warned fallbacks, so they were not converted into findings.

## Metrics

- natural cases detected: `2/2`;
- high-severity natural cases detected: `1/1`;
- exact case predictions: `7/7`;
- clean controls: `5`;
- clean false-positive rate: `0/5 = 0.00`;
- Wilson 95% interval for clean FPR: `[0.00, 0.4345]`;
- frozen evaluator/rule changes after intake: `0`;
- GPU runs, paid APIs, and external disclosures: `0`.

The wide clean-control interval is material: five framework-level controls do
not establish a low ecosystem false-positive rate. P5 supports case-based
external transfer, not prevalence, reliability certification, or a universal
reward-hacking defense.

## Publication consequence

P5 removes the largest natural-transfer blocker for a narrowly framed TMLR
audit/tooling paper. It does not turn ARCA into a main-conference method paper:
P4 has no valid downstream dynamic result, one P5 case is limited to an
evaluation-replay boundary, and the candidate sample is purposive rather than
prevalence-representative.
