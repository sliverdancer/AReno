# X1 pinned external-environment acquisition protocol

Status: `FROZEN_AWAITING_DOWNLOAD_AUTHORIZATION`

The source identities were resolved by read-only `git ls-remote` on 2026-08-02.
No repository, package, task data, or model was downloaded while freezing this
stage.

## Frozen sources

- Tau3: tag `v1.0.1`, commit
  `b711c1ead46f55111bf765cf44d5da8bacc2d28c`, expected MIT license.
- BFCL: tag `v1.3`, commit
  `ea13468e4423454d0c213704fb87cf7cb3990433`, expected Apache-2.0 license.

Tau3 may supply development qualification and real-environment training.
BFCL is sealed external evaluation only. BFCL task or answer content must not be
opened during source installation, adapter development, prompt selection, or
hyperparameter selection.

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
