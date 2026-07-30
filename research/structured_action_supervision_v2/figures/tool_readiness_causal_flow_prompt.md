# Scientific schematic prompt

Create a conference-quality, colorblind-safe left-to-right scientific flowchart
showing two separated phases.

Top phase, labeled "Consumed SAS v1.1 evidence": default Qwen3 thinking plus a
128-token generation ceiling leads to 128 of 128 responses entering thinking
without closing the thinking block, then zero raw or parsed tool calls, then
degenerate reward and collapsed AF/LF masks, ending in KILL.

Bottom phase, labeled "New SAS-TR-v2.0 qualification": compare default-thinking
128/512-token diagnostic cells against non-thinking 128/512-token eligible
cells. Route the eligible cells through strict gates of at least 95 percent
first-call executability and at least 75 percent complete four-turn protocols.
A pass validates the selected cell before a new AF/LF protocol may be planned;
a failure kills the Qwen3-0.6B instrument without automatic model escalation.
Add a prominent note that tool readiness is instrument qualification and not
supervision-efficacy evidence.

Use a clean white background, Okabe-Ito colors, high contrast, sans-serif
typography, and no decorative icons.
