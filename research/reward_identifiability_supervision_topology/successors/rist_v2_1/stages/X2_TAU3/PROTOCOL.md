# X2 Tau3 deterministic environment qualification

Status: `TERMINAL_KILL_X2_GOLD_ACTION_INSTRUMENT`

Tau3 is used only for development qualification and future real-environment
training. Its upstream test split is retired and is not a confirmatory source.
BFCL remains the sealed external evaluation.

For each of airline and retail, development IDs are the first eight
integer-sorted upstream train IDs whose reference trajectory contains at least
one tool action. All other upstream train IDs remain training IDs. The rule is
structural and outcome-blind. Upstream test IDs are excluded from every Tau3
role.

Qualification replays every development reference trajectory twice from a
fresh environment with outbound socket connections disabled. Every action,
raw tool result, chained database-state hash, and terminal transcript hash must
match exactly. Any exception, nondeterminism, split overlap, or network attempt
is `KILL_X2_ENVIRONMENT_QUALIFICATION`.

This is an environment/instrument qualification, not a model result and not
evidence for the supervision-topology hypothesis.

## Terminal execution

The first frozen replay failed before completing replay A: a retail reference
action raised upstream `ValueError: Product not found`. No qualification result
was produced. Under the registered exception rule this protocol is terminal and
must not be selectively rerun. The failure invalidates this gold-action replay
instrument; it is not evidence that Tau3 itself is nondeterministic.
