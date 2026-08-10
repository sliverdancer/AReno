# RIST C0 v4.0 calibration terminal report

Decision: `KILL_C0_V4_CALIBRATION_NO_COMMON_RESOLUTION_CONTRAST`

This is a scientific terminal result, not an infrastructure failure. The v4
semantic-code transport passed capacity canary and full calibration collection
integrity, but calibration did not produce the required cross-model reward
resolution contrast. Qualification, held-out/BFCL access, GPU training, and all
supervision-topology training experiments remain closed.

## Frozen source and runtime

- Source commit used for serving/collection: `16703245e43b5d5a07df411729ac7b30adade075`
- Reward-resolution analyzer commit: `f98c09e2b6c453ebd1cf7e98b5937abf84d8336c`
- GPU: `NVIDIA A800 80GB PCIe`, UUID `GPU-5b7fc2ad-3979-814c-b1fc-0de7714813f5`
- Qwen3-0.6B revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Gemma4 E2B revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- Python realpath: `/root/miniconda3/bin/python3.12`

## Capacity canary

Capacity canary ran first, sequentially by family, with exactly one task and
eight rollout seeds per model. Both families completed with zero retries.

- Final decision: `PASS_CAPACITY_CANARY_COLLECTION_TO_SEPARATE_ANALYSIS`
- Final JSON SHA-256: `98f444aa52bc18796cf95891e9039428a48f00664cef14b396a97804d2fad114`
- Full evidence hash list SHA-256: `286481c5e669d066a7a52838d2597d3d6fb1e14f8bc5b6944d3ab0c0c52e882d`
- Qwen trajectories: `8/8`, raw journal SHA-256 `6f178a8f0ec491e0e86b9a39e1ebe1929863f46552e2dd36f28a37e9554772f0`
- Gemma trajectories: `8/8`, raw journal SHA-256 `4319bb1394f21dd7b7d0b69690da332f6534ea8dfc5adcc3a6d77449591575c7`

## Calibration collection

Calibration then ran sequentially by family with 32 tasks and 32 rollout seeds
per model. Both families completed with zero retries and no infrastructure
error.

- Final decision: `PASS_CALIBRATION_COLLECTION_TO_SEPARATE_ANALYSIS`
- Final JSON SHA-256: `7b322597fbb3e7c0735189e83fbc6163affa1d3552c68acade2139be42ca1240`
- Full evidence hash list SHA-256: `a48603ec510ae8e7c213fa75e9cf3928440548483bf62ec9494065a855bbc300`
- Qwen trajectories: `1024/1024`
- Qwen trajectory artifact SHA-256: `9683d7da6851819144e257022f65b05b3efdd9f6fc039c6efaea7f7745b62e60`
- Qwen raw journal SHA-256: `5918937e7b75d8137ecf4a49bb8b283ccdab803d9c5e7500fb3c550012691b69`
- Gemma trajectories: `1024/1024`
- Gemma trajectory artifact SHA-256: `9eb918b0c9492f151b51176f6f9f920cd9a3fa17af565d681b8a464b96f70bf8`
- Gemma raw journal SHA-256: `973b7f31ed9a8d1626cb94b8618ca2a5a28cd0f51ab52daf5855ba2685698844`

## Reward-resolution analysis

The analyzer was frozen before outcome inspection. It classifies a cell as:

- `high` if at least two of four task groups have mixed rewards;
- `low` if zero of four task groups have mixed rewards;
- `ambiguous` if exactly one task group has mixed rewards.

Results:

- Analysis SHA-256: `9fba8bc7d1dbff14d5a4e2dd9bf87a98083daa39b470cfb258e73f96e6ef6260`
- Admission artifact SHA-256: `8e534fcd56503a2c7692aaca328bb601e5b3c3a0defe9ba7ad6b7e143ef5e34e`
- Common low cells: `c00`, `c02`, `c03`, `c04`, `c05`, `c06`, `c07`
- Common high cells: none
- Qualification accessed: `false`
- Held-out/BFCL accessed: `false`
- Training performed: `false`

The GO criterion required at least two common low cells and two common high
cells. v4 satisfied the low-cell side but had zero common high cells, so it is
terminal for the current C0 route.

## Backup

The full remote evidence bundle was copied to:

`F:\AReno_research_backups\rist_v4\rist_v4_c0_capacity_calibration_1670324_20260810.tgz`

Bundle SHA-256:

`a136900d8f94470f966cbe6c1a7c680048819c28837c05393cdf289dfcd2c533`

## Research implication

v4 fixed the v3.1 long-hash-code pathology enough to show that Gemma can solve
the easiest cell perfectly, but it did not create a shared high-resolution band
under the frozen group-relative criterion. The current RIST C0 instrument remains
unsuitable for opening qualification or training. The paper route should not
claim a positive supervision-topology effect from this lineage.
