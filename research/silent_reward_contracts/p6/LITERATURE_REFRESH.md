# P6 literature refresh

Search date: `2026-08-02`

Scope: direct substitutes for a source-pinned, cross-framework agentic-RL
reward-contract auditor and current publication routes. This refresh extends,
but does not rewrite, the frozen P0 and P5 screens.

## Channels and queries

Three primary-source channels were checked: arXiv metadata, OpenReview records,
and official project or venue pages. Registered query families included
`agentic reinforcement learning audit`, `reward contract language model`,
`reward pipeline agent`, `silent failure reinforcement learning`, and
`training pipeline reward LLM`. Exact title checks were also rerun for the
closest P0/P5 papers and systems.

## Direct-substitute decision

Decision: `NO_DIRECT_SUBSTITUTE_FOUND_P6_REFRESH`

Recent reward-design work such as
[CM2](https://arxiv.org/abs/2602.12268), iterative reward calibration
([arXiv:2604.02869](https://arxiv.org/abs/2604.02869)), and
[PROVE](https://arxiv.org/abs/2606.03892) changes how rewards are constructed.
The [credit-assignment survey](https://arxiv.org/abs/2604.09459) organizes how
credit is distributed. [RHB](https://arxiv.org/abs/2605.02964) and SpecBench
measure agent exploitation of an objective. [Agent2
RL-Bench](https://arxiv.org/abs/2604.10547) evaluates whether an engineering
agent can close an RL loop. None of the screened records combines typed-trace,
reward-flow, advantage-support, treatment-identity, and provenance checks with
cross-framework execution and an unseen-framework evaluation.

The nearest comparator remains framework-native preflight plus Agent2
RL-Bench. Native preflight validates accepted states in one system; Agent2
RL-Bench evaluates an agent that engineers a pipeline. ARCA audits whether a
researcher's supplied pipeline preserved the intended experiment across
interfaces. This is a bounded title/abstract/artifact conclusion, not a proof
that no unpublished or differently named substitute exists.

## Venue evidence

- [TMLR](https://www.jmlr.org/tmlr/) uses rolling submission and emphasizes
  technical correctness over subjective significance. Its official
  [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) ask
  whether claims are accurately supported and whether some readers would be
  interested.
- TMLR requires double-blind manuscripts and anonymous supplementary material
  in its [author guide](https://jmlr.org/tmlr/author-guide.html). Accepted work
  can receive Journal-to-Conference certification according to the official
  [reviewer guide](https://jmlr.org/tmlr/reviewer-guide.html).
- The official [NeurIPS 2026 Evaluations & Datasets
  call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
  explicitly included audits, negative results, protocols, tools, and
  evaluation methodologies. Its May 2026 deadline has passed, so it is venue
  evidence for a future cycle rather than an available 2026 submission.
- The official ICLR site lists a 2027 meeting but did not expose a 2027 call in
  this refresh. No unannounced deadline or unchanged policy is assumed.

## Search limitation

The refresh searched at title, abstract, official documentation, and known
artifact level. It did not perform full-text screening of every 2026 paper.
The novelty claim remains defeasible: discovery of a direct substitute before
submission triggers a new gate.
