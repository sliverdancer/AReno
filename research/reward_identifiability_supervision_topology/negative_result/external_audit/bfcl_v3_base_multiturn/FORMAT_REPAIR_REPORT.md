# BFCL format repair adapter

Status: `CPU_ONLY_FORMAT_REPAIR_READY_FOR_NEW_ONE_REQUEST_CANARY`

Scope:

- CPU-only.
- No model request.
- No API use.
- No GPU use.
- No training.
- No held-out/sealed access.
- No raw response access.

## Repair implemented

The adapter converts public BFCL function-doc schemas from:

```json
{"type": "dict"}
```

to JSON Schema-compatible:

```json
{"type": "object"}
```

The conversion is recursive and preserves property names, required fields,
descriptions, defaults, and task selection.

## Parser fixtures

Synthetic Qwen-format fixtures pass for:

- `<tool_call>{...}</tool_call>`;
- raw JSON object;
- JSON object containing a `tool_calls` array.

The prose-only fixture remains unparsed, which is the desired behavior.

## Frozen next canary

`REPAIRED_FORMAT_CANARY_TEMPLATE.json`

Template SHA-256:

`204fff87b56950b475f3ee7cbdc2e13d765a97e6f70a1d5de2754b95b03bbca8`

The template is not executable until runtime fields are bound and separately
authorized. It still permits only:

- one task;
- one model;
- one rollout;
- zero retries.

The success gate for the repaired canary is parseability, not strict BFCL
success. Full BFCL audit remains closed.
