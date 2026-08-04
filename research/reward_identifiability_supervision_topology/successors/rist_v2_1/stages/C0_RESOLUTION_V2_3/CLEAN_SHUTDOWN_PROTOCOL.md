# RIST C0 v2.3 clean-shutdown successor protocol

Status: `CPU_ONLY_NOT_GPU_VALIDATED`

The consumed capacity run remains immutable. Its two complete eight-response
collections are valid capacity evidence, while its post-collection process-
group termination is recorded as an operational anomaly.

The original `deployment_entrypoint.py` remains byte-for-byte unchanged. The
successor `graceful_deployment_entrypoint.py` reuses the original receipt and
six-identity validation. It changes only launcher supervision: `SIGINT` and
`SIGTERM` received by the supervisor are forwarded to the direct server child.
It must not signal the process group or rollout-worker descendants. Original
signal handlers are restored whether the child exits normally or `wait()`
raises.

CPU tests establish signal topology and preservation of all original
deployment rejection gates. They do not establish that AReno completes a
clean GPU shutdown. A separately authorized, outcome-free GPU serving canary
must start one frozen model through the successor, send no model requests,
signal only the supervisor, and require all of the following:

- server log contains normal application shutdown completion;
- server log contains no `Op.ROLLOUT_SESSION_END` failure;
- the entrypoint and all descendants exit within the frozen timeout; and
- final GPU memory and compute-process count are zero.

Calibration, qualification, held-out/BFCL access, inference requests, and
training remain unauthorized by this protocol.
