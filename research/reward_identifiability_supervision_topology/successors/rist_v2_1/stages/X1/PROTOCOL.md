# X1 pinned external-environment acquisition protocol

Status: `FROZEN_AWAITING_DOWNLOAD_AUTHORIZATION`

The source identities were resolved by read-only `git ls-remote` on 2026-08-02.
No repository, package, task data, or model was downloaded while freezing this
stage.

## Frozen sources

- Tau3: annotated tag `v1.0.1`, tag object
  `b711c1ead46f55111bf765cf44d5da8bacc2d28c`, peeled commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`, expected MIT license.
- BFCL: tag `v1.3`, commit
  `ea13468e4423454d0c213704fb87cf7cb3990433`, expected Apache-2.0 license.

The original CPU freeze recorded the Tau3 annotated tag object as the commit.
Acquisition exposed the distinction before installation or results; the lock
now records both identities and validation requires the tag to peel to the
frozen commit.

Tau3 may supply development qualification and real-environment training.
BFCL is sealed external evaluation only. BFCL task or answer content must not be
opened during source installation, adapter development, prompt selection, or
hyperparameter selection.

Tau3 upstream test is retired from confirmatory use because its loader reads a
unified task file before applying the requested train split. See
`TAU3_UPSTREAM_TEST_RETIREMENT_20260802.md`. This does not affect Tau3's allowed
development/training roles or BFCL's sealed role.

## Authorized-run order

1. Clone each source into a separate temporary directory at the exact commit.
2. Run `validate_checkout.py` before installation.
3. Record source-tree hashes, license text hash, Python version, resolver output,
   and lockfile in an immutable acquisition manifest.
4. Install each source in a separate virtual environment; do not add either as
   an AReno dependency.
5. Run upstream unit tests that do not open benchmark task content.
6. For Tau3 only, freeze development task IDs and run deterministic gold-action
   replay twice from clean reset. Raw state hashes and rewards must match.
7. Stop before BFCL data access, any model endpoint, inference, training, or GPU.

Any commit mismatch, dirty checkout, missing license, nondeterministic Tau3 gold
replay, hidden network call, or task-ID overlap is a fail-closed `KILL_X1`.
