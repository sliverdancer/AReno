# Post-T0b v1.0 route audit

Status: `OBJECTIVE_OPEN_NEXT_ACTION_PUBLIC_API_AUTHORIZATION`

This audit maps the active research objective to direct evidence after terminal
T0b v1.0. A missing item is not inferred from a nearby engineering check.

| Objective component | Direct completion evidence required | Current evidence | Verdict |
|---|---|---|---|
| Stable interaction | two families complete all 8 fresh four-turn trajectories with exact calls and actual runtime token IDs | each family completed one exact first-turn call; the metadata channel then failed | missing |
| Token-matched robustness | seed-level step-matched runs plus common-token endpoint and normalized AUC without material sign reversal | analysis contract only | missing |
| Two model families | at least Qwen and non-Qwen pass T0 and E1, then produce training outcomes | both fixed snapshots and one short call verified; neither passed T0/E1 | missing |
| Two algorithms | GSPO and GRPO produce independently capacity-qualified and completed runs | CPU design only | missing |
| Real tool environment | Tau3 airline policy training with fixed user simulator and upstream non-LLM task reward | environment and adapter CPU-qualified; no policy rollout/training | missing |

## Evidence-quality judgment

The exact first-turn calls are direct evidence that both checkpoints can load,
parse the fresh schema, and execute a short request on the 24 GB 4090D. They are
not evidence of stable four-turn interaction, trainability, treatment validity,
or supervision effects. Treating them as such would be construct
overgeneralization. T0b v1.0 is therefore an infrastructure failure with no
directional scientific result.

## Shortest admissible route

1. Authorize and implement the two-part additive serving metadata change, then
   pass its CPU serialization and compatibility tests.
2. Freeze T0b v1.1 with new tasks/nonces and a new serving commit; separately
   authorize sequential Qwen/Gemma short serving. A 24 GB 4090D is provisionally
   adequate only for this short gate.
3. If T0 passes, run E1 optimizer-step canaries for both GSPO and GRPO. Gemma
   must use a separately authorized larger-memory GPU or be replaced only by a
   newly versioned, fully qualified non-Qwen checkpoint.
4. Run C0 mixed-reward calibration before opening the 48-run diagnostic pilot.
5. Execute the paired 48-run step-matched pilot, estimate seed-level variance,
   and apply the common-token robustness gate. Do not call three seeds
   confirmatory evidence.
6. Freeze prospective seed count and add the third checkpoint required by the
   main-conference matrix before confirmatory execution.
7. Only after the synthetic interaction transports, execute the Tau3 airline
   pilot with a fixed zero-retry user simulator; sealed BFCL remains a later,
   independent authorization.

Passing T0b v1.1 would repair treatment measurement only. It would not by itself
increase the main-conference status beyond `OPEN_NOT_UPGRADED`.
