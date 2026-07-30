# P1 Formalization: Block Risk, Budget Nesting, and On-Policy Refresh

Protocol: `CARE-P1-v0.1`
Scope: CPU-only theorem check and exact-oracle falsification
Status before execution: `OPEN`

## Target and selection family

For behavior-policy iteration \(k\), trajectory \(i\), and turn \(t\), let
\(\hat c_{kit}\) be a frozen cheap credit estimate and \(c_{kit}\) the audited
counterfactual credit. The estimator, confidence score, and deterministic
tie-breaking rule must be fixed without fitting on the calibration labels used
by the same risk-control call.

First construct a confidence-ranked prefix whose total declared gradient-token
mass does not exceed budget \(B\). This prefix is fixed before threshold
selection and is not backfilled. For strictness threshold \(\lambda\), retain
only prefix turns with confidence greater than \(\lambda\). The selected sets
are therefore nested as \(\lambda\) increases.

The primary loss is

\[
L_{ki}(\lambda)=\frac{1}{B}\sum_t m_{kit}\,
\mathbf{1}\{t\text{ selected at }\lambda\}\,
\mathbf{1}\{\operatorname{sign}(\hat c_{kit})
\ne \operatorname{sign}(c_{kit})\},
\]

where \(m_{kit}\) is the declared gradient-token mass. Because selected mass is
at most \(B\), \(L_{ki}\in[0,1]\). It is non-increasing in \(\lambda\).

## Per-iteration block guarantee

Condition on the complete training history before iteration \(k\). Assume:

1. the \(n_k\) calibration trajectories and one future update trajectory are
   exchangeable draws from the same frozen behavior policy and task mixture;
2. each whole trajectory is one block; arbitrary dependence among its turns is
   allowed;
3. \(L_{ki}(\lambda)\) is bounded by one, right-continuous on the finite
   threshold grid, and non-increasing in strictness;
4. the scorer, budget prefix, and all tuning choices are fixed independently of
   these calibration labels.

Choose the most permissive threshold satisfying

\[
\frac{\sum_{i=1}^{n_k}L_{ki}(\lambda)+1}{n_k+1}\le\alpha.
\]

The conformal-risk-control theorem then gives, conditionally on the previous
training history,

\[
\mathbb{E}[L_{k,n_k+1}(\hat\lambda_k)\mid\mathcal H_k]\le\alpha.
\]

Taking expectations again yields the same marginal per-iteration bound. If the
conditions hold at every iteration and calibration is refreshed before each
policy update, linearity of expectation also controls the average loss across
a fixed number of iterations. This is a direct conditional application of
conformal risk control, not a claim of uniform validity after arbitrary policy
shift.

## What is and is not guaranteed

Guaranteed target:

- expected wrong-sign gradient mass per trajectory, normalized by the frozen
  budget.

Not guaranteed:

- wrong-sign fraction conditional on selection;
- high-probability risk for every realized calibration split;
- correctness after carrying a threshold to a changed policy or task mixture;
- downstream task improvement;
- lower FLOPs from applying a loss mask.

The conditional selected-error fraction is non-monotone in general. It remains
a reported diagnostic and cannot be advertised as the conformal theorem's
target.

## Exact controlled bifurcation

`bifurcation_oracle.py` defines a four-decision deterministic MDP. The terminal
reward depends on weighted binary decisions. For each recorded state, the
oracle enumerates both current actions and all future continuations under a
frozen uniform policy, then computes

\[
c_t=Q(s_t,a_t)-\tfrac12[Q(s_t,0)+Q(s_t,1)].
\]

All \(2^4\) trajectories and all 64 turn labels are enumerable without a model
or GPU. The required mechanism check is that successful trajectories can
contain negative-credit turns and failed trajectories can contain
positive-credit turns; a terminal reward broadcast cannot recover this fact.

## Analytic stress tests

The artifact generator uses exact binomial sums, not Monte Carlo:

1. **Independent turns:** ordinary turn-level CRC is valid when the turns
   really are independent units.
2. **Clustered turns:** duplicating one trajectory-level error across ten turns
   and pretending they are ten independent calibration units can violate the
   nominal expected risk.
3. **Policy shift:** a threshold calibrated at a low bad-trajectory rate can
   violate the target after the rate changes; recalibrating on blocks from the
   changed policy restores the same-iteration bound, often by abstaining more.

These tests establish a need for block units and refresh. They do not by
themselves establish sufficient algorithmic novelty for a main-conference
paper.
