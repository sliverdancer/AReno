# RIST C0 v3.1 calibration terminal report

Date: 2026-08-10

Decision: `KILL_CURRENT_C0_V3_1_TRANSPORT_DO_NOT_OPEN_QUALIFICATION`

## Execution

- Remote GPU: NVIDIA A800 80GB PCIe, UUID `GPU-5b7fc2ad-3979-814c-b1fc-0de7714813f5`.
- Source commit: `ab271edb1fde9048795364ca6be77171541034a9`.
- Bound calibration manifest SHA-256: `6bb464ad06e1f001a0d4bdf71d1ce344dad2b644f55465a87122de0019bfdf3e`.
- Qwen3 revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Gemma4 revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- Qwen receipt artifact SHA-256: `03db54e40abd327ad8b0557fcb1a79884ff542e25665a37fba48cd72970ea739`.
- Gemma receipt artifact SHA-256: `11cff721924d77c87a0c33f40e75c179d29515d82aa33e925067d27cc8abaec4`.

## Collection validation

Both jobs completed with zero retry and passed the v3 validator/finalizer.

| family | trajectories | raw responses | trajectory SHA-256 | journal SHA-256 |
|---|---:|---:|---|---|
| qwen3 | 1024 | 1024 | `3efb68a61ae520176efda5bfaac5b5adb3c7b1e6e79b8b19232a73b76137ca37` | `bf3455f4fcce51ff1e674e63e09506e7b01e15f4735920a001a6865ad29d52ca` |
| gemma4 | 1024 | 1024 | `482e012be08775b7505dc0ace6d1847eb305d5eadad136adecb66534583015cb` | `cc1f9789a9c30bb6b541580825d873a67dff765514db153410214c58582caad8` |

Finalizer decision: `PASS_CALIBRATION_COLLECTION_TO_SEPARATE_ANALYSIS`.

## Calibration resolution result

Calibration-only resolution analysis produced:

- common low cells: `c00`, `c01`, `c02`, `c03`, `c04`, `c05`, `c06`, `c07`;
- common high cells: none;
- qualification gate: `false`.

This fails the pre-registered requirement of at least two common low cells and
at least two common high cells across both model families. Qualification remains
closed. Held-out and BFCL were not accessed. No training was performed.

## Evidence backup

Raw evidence is intentionally not committed to Git. It is backed up locally at:

`F:\AReno_research_backups\rist_v3\calibration_ab271ed\areno_rist_v3_calibration_ab271ed.tgz`

Evidence package SHA-256:

`c81dfbf35ce914ee052617506e5ee129abc53cecd640fce23fc6769a5d034d2f`

Resolution analysis SHA-256:

`6b5430dabfb57cf315fd99e04f4d86e25e32a4d32c8a4b299e187da25683eaee`

## Interpretation

The current v3 synthetic transport is too hard or too collapsed for the selected
models: all eight shared cells are low-resolution, and there is no high-resolution
contrast to support the planned group-relative supervision-topology experiment.
The correct next step is CPU-only instrument redesign, not qualification or
training.

