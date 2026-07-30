# Q0 Instrument Qualification Report

Decision: `BLOCKED_PUBLIC_SEED_CONFIG_AUTHORIZATION`

Q0 completed every currently authorized deterministic check:

- four factorial masks are distinct;
- the all-tool-call `final_answer` control has zero trainable tokens;
- an explicit zero-signal batch now skips the backend optimizer step;
- missing, multiple, wrong-name, malformed-JSON, non-object, and schema-invalid
  calls are rejected without synthesis;
- raw model arguments are retained byte-for-byte;
- 80 valid task constraints are split 48/16/16 with no repeated signature;
- the strict reward requires the exact four-call protocol;
- the outcome and resource metric schema is frozen.

Verification:

- 89 targeted agentic/research tests passed;
- 382 CPU tests passed when the optional serving test module was excluded;
- full CPU collection could not start because this virtual environment lacks
  `fastapi`;
- syntax compilation and `git diff --check` passed.

Q0 cannot pass because stochastic training seed provenance does not exist on
the public train surface. Adding it requires explicit approval under
`AGENTS.md`. Q1 remains unopened, and no GPU/model work has occurred.
