# P6 reviewer and submission gate

Protocol: `ARCA-P6-PAPER-v0.1`

Decision date: `2026-08-02`

Decision: `PASS_P6_TMLR_READY`

## Answer first

The complete anonymous paper, source-backed claim audit, deterministic
anonymous artifact, and reviewer gate pass the frozen P6 criteria. This opens a
TMLR submission-preparation route. It does not authorize submission, public
artifact release, maintainer contact, or GPU execution.

## Manuscript evidence

- manuscript: `paper/main.tex`;
- compiled PDF: `paper/main.pdf`, 8 pages;
- official TMLR style revision:
  `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`, with file hashes in
  `paper/style_manifest.json`;
- compilation: `pdflatex -> bibtex -> pdflatex -> pdflatex`, exit code 0;
- final log: no undefined citation, undefined reference, overfull box, or fatal
  error; one non-blocking underfull page box remained;
- PDF visual audit: all eight rendered pages inspected, with no clipped text,
  overlap, broken table, unreadable glyph, or placeholder;
- PDF anonymity scan: no email, local `/mnt` or `/home` path, SSH command,
  configured identity, or user-specific string.

## Artifact evidence

- archive: `artifact/dist/ARCA-anonymous-artifact-v0.1.tar.gz`;
- independent deterministic builds: byte-identical under the CPU test;
- structured audit: `artifact_audit.json`;
- allowlist, SHA-256, JSON/CSV syntax, credential, identity, local-path, and Git
  metadata checks: pass;
- invalid P4 evidence archive: excluded; its redacted terminal gate record is
  retained.

## Reviewer passes

Round 1 returned a desk-reject signal because the nearest comparator and
novelty delta were underspecified and eight citations were stacked in one
sentence. The paper was revised without changing results or thresholds.

Round 2 returned `Pass to Review`, score `7.6/10`, with zero major issues and
two moderate advisories. The numeric-consistency advisory was resolved by the
machine-readable `paper_claim_audit.json`. The remaining novelty advisory was
low-confidence, had no verified quote, and was addressed by naming native
preflight plus Agent2 RL-Bench as the nearest comparators and freezing the P6
literature refresh. It remains a normal review risk, not an unsupported central
claim.

The final deterministic gate reports `裁定：通过`: anonymous submission,
figures/tables, notation, acronym definitions, placeholders, bibliography, and
references all meet the blocking checklist. Script advisories about labels
being referenced before their definitions are standard forward references and
compiled correctly. The abstract remains one paragraph because the official
TMLR template requires it.

## Claim boundary

The permitted contribution is a case-based, cross-framework reward-contract
audit and reproducible tooling artifact. The paper does not claim ecosystem
prevalence, a low universal false-positive rate, improved model quality, a new
RL method, or a valid P4 result. Any public submission or new GPU study remains
a separate authorization gate.
