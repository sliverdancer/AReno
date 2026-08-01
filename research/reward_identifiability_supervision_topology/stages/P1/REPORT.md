# P1 CPU Task and Measurement Freeze

Decision: `PASS_P1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`

The research-only workflow generator satisfies all eight frozen gates on both
the 96-task train development split and the 32-task qualification merge gate.
No model, API, serving process, or GPU was used.

## Frozen results

| Check | Train | Qualification | Result |
| --- | ---: | ---: | --- |
| Tasks | 96 | 32 | pass |
| Balanced factorial cells | 32 | 32 | pass |
| Evaluator score | 8/8 | 8/8 | pass |
| Possible action sequences | 1 to 7,776 | 1 to 7,776 | pass |
| Mixed-group probability, G=8 | 0.000 to 0.961 | 0.000 to 0.961 | pass |
| Reward-resolution strata | low/intermediate/high | low/intermediate/high | pass |
| Byte-identical regeneration | yes | yes | pass |
| Cross-split signature leakage | zero | zero | pass |

The qualification split contains 20 low-resolution, 3 intermediate-resolution,
and 9 high-resolution analytic tasks. These labels come from a uniform policy
over explicit action choices. They are construction diagnostics and must not be
presented as language-model behavior.

## Key scientific property

The instrument contains cells with at least 256 distinct admissible action
sequences but less than 0.2 probability of a mixed binary-reward group at eight
samples. It therefore separates action diversity from reward resolution by
construction and can test the mechanism suggested by the terminal B3 result.

## Arbor result

Initial material `7f4c8b5` scored 8/8 on train and 8/8 on qualification. The
bounded evaluator cannot exceed eight. Arbor therefore accepted M0 and stopped
at cycle 0 rather than inventing changes to a saturated objective. No executor
branch or scientific held-out score was used.

## Provenance

| File | SHA-256 |
| --- | --- |
| train | `8487a3587cc634409d9465af2c091982ccad8f6f4426007e96268513cdab3ded` |
| qualification | `8018137606e12da0f0096ac86f11312d94d631326965198783ebf9cecc94570f` |
| held-out | `a318c50d7036abf8f58922730b7ba9b19324679bf1ee20459842cd742756c756` |
| manifest | `80eb3bbe9ae9ae23ade34f724233d7c61919bd183b7dc3dd4ba4a93ccd0b14f1` |

Held-out was generated and content-addressed. It was not scored or used to
choose a candidate.

## Claim boundary and next gate

P1 proves only deterministic task construction and analytic measurement
coverage. It does not show that any checkpoint produces the intended reward
strata and does not estimate AF/LF/AN/LN effects.

The main-conference hook returns
`STAY_DIAGNOSTIC_OPEN_P2_GPU_AUTHORIZATION`. P2 must run inference-only
qualification on at least two checkpoints from two model families before any
factorial training can open.

