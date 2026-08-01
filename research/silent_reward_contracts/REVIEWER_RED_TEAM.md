# Reviewer red-team

## Likely major objections

1. **The evaluator detects its own mutations.** Correct. The held-out score is
   mutation-transfer evidence only. Countermeasure: lead with natural cases,
   publish mutation operators, and keep OpenRLHF clean results explicit.
2. **Five systems with cases do not establish prevalence.** Correct. P5 used a
   purposive sample and must be reported as case-based external validation.
3. **Rules may be obvious assertions.** The contribution must be the executable
   cross-boundary contract, provenance, and demonstrated conclusion flips—not
   novelty claims about JSON parsing or finite checks individually.
4. **The AReaL case was found after held-out evaluation.** Report it only as
   post-freeze replication. It cannot improve the held-out score.
5. **ARCA may overfit AReno/veRL.** Frozen OpenRLHF clean controls constrain
   false positives, but only more natural unseen cases can answer this fully.
6. **No downstream learning result.** P4 measures dynamic reward consequence,
   not learning efficacy. The paper must remain an audit paper unless a later,
   separately frozen study justifies a stronger claim.
7. **The CARe failure is self-inflicted.** Treat it as a discovery incident,
   preserve the terminal result, and rely on independent veRL/AReaL evidence.
8. **Native framework checks may already suffice.** On registered OpenRLHF
   mutations, native preflight detects one of four families. Verify baseline
   definitions and avoid implying maintainers intended ARCA's scope.
9. **The slime case is not a core training failure.** Correct. It is an
   official evaluation-replay boundary and must be labelled as such in the
   abstract, table, and limitations.
10. **Zero false positives on five controls is weak.** Correct. Report the
    Wilson interval `[0.00, 0.4345]`; do not turn the point estimate into an
    ecosystem reliability claim.
11. **rLLM's fallback may be called user error.** Show that lightweight dict
    returns are documented, execute both coercion implementations, retain the
    explicit-reward positive control, and frame the contribution as contract
    ambiguity rather than maintainer negligence.

## Submission blockers

- any wording that calls mutation recall natural bug recall;
- any aggregation of seeds as independent frameworks;
- any hidden repair or rerun after a consumed gate;
- missing source hashes, raw artifacts, or exact environment commands;
- claiming training improvement from P4's one-step reward restoration;
- public disclosure before maintainer-contact authorization.
