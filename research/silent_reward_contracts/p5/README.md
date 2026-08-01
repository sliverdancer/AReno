# P5: external natural validation

P5 is a CPU-only, source-pinned external validation of the frozen ARCA rules.
Read `FREEZE.md` before inspecting candidates. Machine-readable outputs will
be written under `artifacts/`; upstream clones and dependency caches remain
outside the repository.

This stage neither reruns P4 nor starts GPU training. It also does not contact
maintainers, open issues, or publish suspected defects.
