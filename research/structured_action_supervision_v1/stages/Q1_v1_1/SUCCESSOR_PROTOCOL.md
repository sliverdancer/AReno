# SAS Q1 Successor Protocol

Protocol: `SAS-P0-v1.1`  
Predecessor: `SAS-P0-v1.0` (`KILL_CURRENT_INSTRUMENT`)  
Status: `PREPARED_FIX_VALIDATION`

## Reason for a new version

The v1.0 attempt failed before its first model request because the public
dynamic agent loader executed a dataclass-bearing module without registering
it in `sys.modules`. No trajectory, reward, optimizer step, or held-out outcome
was consumed. The v1.0 result and `KILL_MAIN_TRACK` assessment remain immutable.

The user explicitly authorized automatic continuation on 2026-07-30. The
successor is therefore a new protocol version rather than a selective
continuation of the failed attempt.

## Sole permitted implementation change

The public `load_agent_run_fn` implementation may:

1. assign a deterministic, collision-safe module name derived from the absolute
   agent-file path;
2. register the module in `sys.modules` before `exec_module`;
3. remove that exact registration if module execution fails.

A CPU regression must load an agent file containing
`@dataclass(frozen=True, slots=True)` through the public loader. No masking,
reward, dataset, sampling, optimizer, model, seed, or step setting may change.

## Frozen experiment

The successor repeats the original qualification-only matrix:

- model: `Qwen/Qwen3-0.6B` from ModelScope;
- algorithm: GSPO;
- arms: AF (`all_assistant`) and LF (`last_assistant`);
- seeds: 1101 and 2202;
- maximum steps: 8 per cell;
- qualification dataset SHA-256:
  `a4a54821febebc60f47f8bb1ca67183c35941ffbc1f2b9bfd5412fe869b92d32`;
- held-out outcomes: unopened.

The original 28,800-second aggregate GPU cap remains in force. For a
conservative accounting, v1.1 receives at most 28,774.619931 seconds after
subtracting the v1.0 controller's 25.380069 seconds.

## Gates

Development gate:

- focused dataclass-loader regression passes;
- SAS and public CLI CPU tests pass;
- CUDA extension still imports on the remote host.

Fresh integration merge gate:

- all four successor commands return zero within the remaining cap;
- every run emits raw trajectories and TensorBoard metrics;
- no OOM, non-finite optimization, fabricated calls, or silent repair;
- AF and LF emit distinct trainable-token masks;
- rewards and resource measurements are non-degenerate.

Failure closes v1.1 without repairing or selectively rerunning a consumed cell.
