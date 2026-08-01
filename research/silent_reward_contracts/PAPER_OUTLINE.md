# Paper outline and analysis plan

## Abstract contract

State the problem, four natural cases in three pinned systems, the frozen
OpenRLHF mutation-transfer result with its clean-control denominator, and the
fact that dynamic GPU consequence is pending. Do not advertise model gains.

## Sections

1. **Introduction** — successful execution is weaker than scientific
   learnability; define a silent contract failure.
2. **Contract taxonomy** — F1–F6 and the trace → reward → advantage → mask →
   artifact chain.
3. **ARCA** — executable rules, positive controls, fail-closed artifacts, and
   runtime cost.
4. **Study design** — discovery/development/held-out split, single-fault
   mutations, source pins, frozen hashes, baselines, and gates.
5. **Natural case studies** — AReno, veRL reward flow, veRL timeout provenance,
   and post-freeze AReaL replication.
6. **Frozen transfer evaluation** — per-family recall, clean false positives,
   baselines, and sensitivity analysis.
7. **Dynamic consequence** — include only if P4 passes; otherwise report it as
   unopened future work.
8. **Limitations and responsible disclosure** — no prevalence sampling, no
   general reward-hacking defense, mutation dependence, and disclosure policy.

## Primary tables and figures

- Figure 1: contract chain and insertion points;
- Table 1: taxonomy, observable symptom, positive control, severity;
- Table 2: natural cases with source commit and conclusion flip;
- Figure 2: held-out macro recall and clean FPR by baseline, with denominators;
- Table 3: per-rule Wilson intervals and framework/workload cells;
- optional Figure 3: paired strict/canonical P4 reward traces by seed.

## Statistical rules

- Framework × workload × family is the inferential cell; fixture seeds are
  repeated measurements.
- Report macro recall, clean false-positive rate, Wilson 95% intervals, and the
  preregistered hierarchical bootstrap.
- For P4, show all six raw step rows and paired differences. With only three
  seeds, use the mechanical informativeness gate and descriptive intervals;
  do not claim significance from an underpowered sign test.
- Report all natural and synthetic denominators separately. Missing or failed
  runs are terminal outcomes, never silently excluded.

## Appendix

Include the direct-substitute matrix, full mutation registry, source and
artifact hashes, exact CPU/GPU commands, terminal CARe history, Arbor tree,
claim ledger, and reviewer red-team.
