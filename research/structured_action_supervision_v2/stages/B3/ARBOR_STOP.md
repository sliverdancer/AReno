# Arbor stop record

The Arbor workflow was selected to keep autonomous experimentation
falsification-first. Its baseline material was commit `57ba60c`, and its
objective was arm-symmetric instrument validity rather than maximizing an
AF/LF outcome difference.

The first frozen evaluator was the B3-A within-group signal gate. It returned
`KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL` before any trainable candidate
existed. No hypothesis-tree search, candidate refinement, held-out merge gate,
or factorial training was opened. Continuing Arbor optimization by changing
sampling, reward, task selection, algorithm, or model would violate the
consumed protocol and requires a new research plan.
