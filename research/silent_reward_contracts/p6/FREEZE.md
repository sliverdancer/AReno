# P6 paper, anonymous-artifact, and reviewer-gate freeze

Protocol: `ARCA-P6-PAPER-v0.1`

Freeze date: `2026-08-02`

State: `FROZEN_BEFORE_MANUSCRIPT_DRAFT`

P6 converts the already frozen P0--P5 evidence into an anonymous paper and a
CPU-reproducible artifact.  It does not reopen P4, generate new scientific
outcomes, or reinterpret failed runs.  No GPU training, model download, paid
API, maintainer contact, public issue, or disclosure is authorized by P6.

## Evidence boundary

The manuscript may claim only the following:

- six natural cases across five source-pinned systems;
- 163 development cases, including three natural cases, with the registered
  development metrics;
- a one-shot OpenRLHF held-out set of 40 source-derived clean controls and 40
  synthetic single-fault mutations;
- two post-freeze P5 natural cases among five purposively selected external
  systems, with five framework-level clean controls and a Wilson 95% clean-FPR
  interval of `[0.0000, 0.4345]`;
- P4-v0.2 was invalid before model loading and produced no scientific result.

Natural cases, source-derived clean controls, synthetic mutations, and
post-freeze replications must always have separate denominators.  The paper
must not claim ecosystem prevalence, model-quality improvement, reward-hacking
prevention, or a main-conference learning-method contribution.

## Anonymous-artifact gate

The artifact passes only if all conditions below hold:

1. a machine-readable allowlist and SHA-256 manifest cover every distributed
   file;
2. two independent builds are byte-identical;
3. the archive contains no Git metadata, author identity, email address,
   absolute local path, SSH endpoint, credential-like token, or invalid P4
   evidence archive;
4. all shipped JSON parses, all shipped CSV has a header, and every manifest
   hash verifies;
5. the artifact's CPU reproduction and audit tests pass in a clean extraction.

## Manuscript gate

The anonymous manuscript passes only if it compiles without an undefined
reference or citation, every quantitative statement maps to a shipped source
artifact, P4 is labelled `NOT_EVALUATED`, limitations include the purposive
sample and wide external-control interval, and the PDF contains no identity or
local path.

## Reviewer gate

Run one full reviewer-style audit, revise only documented blockers, and run a
second gate.  The P6 publication state is one of:

- `PASS_P6_TMLR_READY`: no critical blocker and no unsupported central claim;
- `CONDITIONAL_P6_REVISION`: the paper is scientifically viable but has one or
  more repairable major blockers;
- `KILL_P6_PAPER`: a direct substitute or unrecoverable validity failure
  invalidates the central audit contribution.

## Dynamic venue policy

Dynamic planning may change the target venue but never the evidence.  TMLR is
the default target.  A main-conference route may open only through a separately
frozen extension that supplies at least one of:

- a broader benchmark contribution with genuinely independent external cases
  and controls;
- a validated systems intervention with measured prevention benefit and
  bounded overhead;
- a new learning method with preregistered downstream evidence.

Without such an extension, main-track method claims remain closed.  Any GPU
extension requires a new explicit user authorization.
