# P3-v0.2 gate figure provenance and review

Asset: `care_p3_v02_gate.png`

Purpose: protocol navigation only. The image is not experimental evidence.

## Generation prompt

Create a publication-quality scientific workflow infographic for an AI
reinforcement learning research protocol, landscape 16:9, white background,
colorblind-safe blue/orange/red/gray palette, clean vector-like flat design,
no logos, no decorative AI imagery, no watermark. Title: "CARe P3
Qualification and Scientific Gate". Left-to-right flow with seven stages:
P3-v0.1 frozen; Loader crash with "first run; 5 unopened"; Evidence preserved;
CPU postmortem; P3-v0.2 regression gate with "Python 3.10 / 3.12"; Explicit GPU
authorization; six-run qualification followed by executable/non-degenerate
gate and PASS to P4 or STOP/KILL P3. Bottom annotation: "No outcome-adaptive
repair • same arms, seeds, assets, and hyperparameters".

## Review

- Text is legible and the branch structure matches the protocol.
- Colors supplement rather than replace text labels.
- The figure correctly separates CPU requalification from GPU authorization.
- It does not display results or imply that P3-v0.2 has run.
- The generated wording was manually checked against
  `POSTMORTEM_AND_PROTOCOL.md`.

The preferred specialized scientific-schematic backend was unavailable because
its API credential was not configured. The repository records this image-tool
fallback rather than requesting or storing a secret.
