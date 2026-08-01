# Arbor hypothesis-refinement report

Objective: maximize held-out detection recall for silent agentic-RL contract
failures while keeping clean-fixture false-positive rate at or below 5%.

Best node: `n2`, semantic trajectory and reward invariants. Best held-out score:
`1.0`; clean false-positive rate: `0.0`. The run began without an implemented
ARCA evaluator, so no comparable initial score is claimed.

The refinement established that typed parsing alone was too narrow. AReno
supported the typed-boundary hypothesis, but its exact subtype did not recur in
veRL (`n4`, dev score 0). The stronger branch covered semantic namespace flow
and structured failure provenance: it detected veRL's reward-key mismatch and
timeout-to-zero conflation and transferred to registered OpenRLHF mutations.
The identifiability branch retained declared treatment aliases and explicit
group-variance preconditions.

There was no observed dev/test metric gap on registered faults, but the data
regimes differ: development includes three natural cases; held-out contains
source-derived clean controls and synthetic faults only. That domain gap is the
main unresolved threat. AReaL's post-freeze natural replication reduces, but
does not eliminate, the threat.

The next admissible refinement is the frozen P4 dynamic intervention. It is
prepared but remains unopened until explicit GPU authorization.
