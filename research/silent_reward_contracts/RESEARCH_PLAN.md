# Research and publication plan

Working title: **Executable Is Not Learnable: Auditing Silent Reward-Contract
Failures in Agentic Reinforcement Learning**

Primary venue: TMLR. A main-conference submission is conditional on stronger
natural external validation and a convincing downstream consequence study;
the current CPU package alone is better framed as an empirical audit/tooling
paper than a method paper.

## Contribution package

1. a six-family executable contract taxonomy spanning trace, reward,
   advantage, treatment, and provenance boundaries;
2. production-path natural cases in three pinned systems;
3. a frozen, CPU-verifiable auditor with clean controls, registered mutations,
   source hashes, and one-shot held-out evaluation;
4. a bounded dynamic intervention showing what ordinary successful execution
   fails to reveal, if P4 passes.

## Completed stages

- P0 literature/direct-substitute screen: passed;
- P1 AReno production-boundary reproduction: passed;
- P2 natural cross-system gate: passed with three development failures across
  two systems;
- P3 frozen held-out transfer: passed, plus one AReaL post-freeze replication;
- reviewer red-team and claim ledger: completed;
- GPU harness: CPU-prepared and tested, not executed.

## Remaining stages

### P4 — bounded dynamic consequence

Execute `ARCA-P4-DYNAMIC-v0.1` once on a newly rented GPU after authorization.
The intervention normalizes string/object tool arguments; it does not change
the model, prompts, data, optimizer, or selected trainable tokens. A pass shows
that the contract changes the observed reward signal in the live pipeline. A
kill ends dynamic escalation and the paper remains CPU/source-audit only.

### P5 — natural external validation

Open only if P4 passes. Solicit or independently reproduce at least two
maintainer-confirmed cases outside the development systems, with disclosure
authorization handled separately. Freeze case inclusion criteria before
contacting maintainers. No public issue or PR may be opened automatically.

Main-conference GO requires at least two new natural cases, at least one in an
unseen framework, and no rule changes after case intake. Otherwise target TMLR
with a narrower audit claim.

### P6 — paper and artifact release

Draft the paper around failure contracts and scientific validity, not model
performance. Release source pins, fixtures, mutation operators, frozen hashes,
artifact schemas, negative controls, and all terminal outcomes. Run an
anonymous artifact audit and a separate paper-review gate before submission.

## Decision schedule

| Gate | GO | KILL / fallback |
|---|---|---|
| P4 | canonical arm restores informative reward | retain CPU audit paper; no rerun |
| P5 | two new natural cases, one unseen system | TMLR narrow claim |
| Paper audit | no critical validity blocker | revise before submission |
| Main-conference positioning | natural breadth plus downstream consequence | TMLR first |

The current strongest defensible route is TMLR. “Main-conference method paper”
remains an aspiration contingent on P4–P5 evidence, not an achieved status.
