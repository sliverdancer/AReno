# Reviewer red-team

## Likely major objections

1. **The evaluator detects its own mutations.** Correct. The held-out score is
   mutation-transfer evidence only. Countermeasure: lead with natural cases,
   publish mutation operators, and keep OpenRLHF clean results explicit.
2. **Three frameworks do not establish prevalence.** Correct. Avoid rates and
   ecosystem-wide language; P5 is required for broader claims.
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

## Submission blockers

- any wording that calls mutation recall natural bug recall;
- any aggregation of seeds as independent frameworks;
- any hidden repair or rerun after a consumed gate;
- missing source hashes, raw artifacts, or exact environment commands;
- claiming training improvement from P4's one-step reward restoration;
- public disclosure before maintainer-contact authorization.
