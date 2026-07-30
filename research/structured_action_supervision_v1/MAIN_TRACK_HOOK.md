# Mandatory Main-Track Upgrade Hook

Every stage closure must contain both:

1. `stage_result.json` — the scientific/engineering gate result; and
2. `main_track_assessment.json` — output of `stage_completion_hook.py`.

Run:

```bash
python3 research/structured_action_supervision_v1/stage_completion_hook.py \
  research/structured_action_supervision_v1/stages/<STAGE>/stage_result.json
```

The hook is deterministic and refuses to overwrite a different existing
assessment. A later stage must not open unless the previous stage has both
files and the assessment hash matches the stage-result bytes.

## Decisions

- `GO_MAIN_TRACK`: allowed only after a valid Q3 confirmatory result and only
  when every frozen main-track criterion is true.
- `STAY_DIAGNOSTIC`: continue, pause, or publish as a bounded diagnostic; do
  not claim a main-track contribution.
- `KILL_MAIN_TRACK`: a terminal invalid/KILL result prevents upgrade on this
  protocol version.

## Frozen main-track criteria

The Q3 result must demonstrate all of:

- valid protocol execution;
- a practically meaningful primary effect whose interval excludes zero;
- completed multi-environment, multi-model, and multi-algorithm evaluation;
- completed direct-baseline comparisons;
- sufficient independent training seeds;
- reproducible artifacts;
- a new method or benchmark contribution beyond static masking;
- no material protocol deviation.

Passing Q0, Q1, or Q2 can never by itself return `GO_MAIN_TRACK`. This prevents
instrument success or a promising pilot from being promoted into a publication
claim.

