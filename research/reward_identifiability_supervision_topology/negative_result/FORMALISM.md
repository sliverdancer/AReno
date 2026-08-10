# Reward-resolution estimability formalism

## Setup

For a task instance \(x\), a policy samples a group of \(K\) rollouts
\(\tau_{1:K}\). A strict evaluator returns binary rewards
\(r_i \in \{0, 1\}\). Group-relative objectives use a normalized advantage
\(A_i = g(r_i, r_{1:K})\), commonly centered within the group.

For a supervision topology \(s\), the training signal on an action span is useful
only if the group contains reward contrast that can induce non-zero advantages.

## Effective reward contrast

Define a rollout group as mixed if:

\[
0 < \sum_{i=1}^{K} r_i < K.
\]

Define cell-level reward resolution as the number of mixed task groups inside a
structural cell:

\[
M(c) = \sum_{x \in c} \mathbf{1}[0 < \sum_i r_{x,i} < K].
\]

In RIST v4.0, each cell contains four task groups and each group has 32 rollout
seeds. The frozen rule was:

- `high`: \(M(c) \ge 2\)
- `low`: \(M(c) = 0\)
- `ambiguous`: \(M(c) = 1\)

For cross-model transport, a cell is admissible only if both model families have
the same high/low label. The C0 GO condition required at least two common low
cells and two common high cells.

## Collapse proposition

If all rewards in a group are identical, then any centered group-relative
advantage is zero for every rollout:

\[
r_1 = \cdots = r_K \Rightarrow r_i - \bar{r} = 0.
\]

The same condition also eliminates rank or z-score contrast because every
rollout is tied. Therefore strict task success provides no group-relative
learning signal for distinguishing which action spans should be supervised.

This does not prove that the task is impossible, or that action-span supervision
cannot matter elsewhere. It only establishes that, under the observed group and
reward design, the topology effect is not estimable from strict terminal reward.

## Diagnostic estimands

The paper uses the following pre-training diagnostics:

- mixed reward group count;
- non-zero-advantage group rate;
- all-fail collapse rate;
- all-pass collapse rate;
- common high/low cell count across model families;
- cross-model transportability of reward-resolution labels.

These diagnostics must be reported before any topology-training experiment. If
the diagnostic fails, the correct action is to stop or redesign under a new
pre-registered protocol, not to train and interpret null effects.
