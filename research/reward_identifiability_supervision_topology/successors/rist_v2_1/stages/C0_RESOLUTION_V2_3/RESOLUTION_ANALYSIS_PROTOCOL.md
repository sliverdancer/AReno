# C0 v2.3 calibration-to-qualification resolution protocol

Status: `CPU_IMPLEMENTED_OUTCOMES_UNOPENED`

`analyze_calibration.py` begins only after the content-blind calibration final
passes. It exclusively creates an outcome-access receipt before interpreting a
response. It replays every raw response with the frozen strict evaluator and
requires exact agreement with the submitted trajectories. It never opens the
qualification task file.

Seeds are sorted into four groups of eight. A group is mixed when both strict
rewards occur. Each structural cell contains four tasks and therefore sixteen
groups. The 95% Wilson rule is frozen:

- collapsed: upper bound <= 0.25;
- resolved: lower bound >= 0.50;
- transition: otherwise;
- heterogeneous: a collapsed/resolved aggregate with fewer than three agreeing
  tasks.

Only cells with the same collapsed/resolved label in both model families enter
the calibration candidate map. At least two low and two high common cells are
required. Otherwise C0 v2.3 is KILL before qualification.

`analyze_qualification.py` consumes the exact calibration admission and the
separately collected qualification split. It tests only the preselected cells;
new cells cannot be admitted even if their qualification outcomes are favorable.
Each family and the cross-family map again require at least two transported low
and two transported high cells. PASS opens E1 capacity; failure terminates the
current main-conference causal route before training.

The first post-receipt error is terminal. Neither outcome directory may be
reused, repaired, or selectively rerun.
