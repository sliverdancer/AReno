# RIST T0b v1.1 execution report

Status: `TERMINAL_FAIL_CLOSED`

The exact `2cca319` Git archive, fresh v1.1 task hash, and both locked model
weight hashes passed verification. No checkpoint was downloaded or replaced.
Qwen3-0.6B and Gemma4 E2B were served sequentially with native attention,
eager decode, thinking disabled, and one running prompt.

The first Qwen request returned HTTP 500 before producing a runtime row because
the exact source archive did not contain the compiled `areno_accel` extension.
The zero-retry client terminated the cell immediately. Read-only comparison
then established that the previously compiled extension came from identical
`areno/accel` sources and the same Python, PyTorch, and CUDA ABI. Linking that
dependency allowed its import without changing tracked `2cca319` source. The
Qwen cell remained consumed and was not retried.

The independent Gemma cell then completed all 32 rows across eight four-turn
tasks with zero retries. Its journal SHA256 is
`dc9bd4012f1523e88ccbae6d534e3f6403037ec016e8b3323e85f6c745c722ba`.
The frozen production name-only mask nevertheless failed all 32 rows: Gemma
response tokens decode to tokenizer-native `call:scan_registry{...}` syntax,
while the frozen mask only recognizes JSON `"name"` fields. Canonical JSON
fixtures therefore did not qualify the real treatment.

Qwen served for 369 seconds and Gemma for 295 seconds, totaling 664 of the
authorized 1,800 seconds. Both servers stopped and the final GPU process list
was empty. No training, held-out access, BFCL content access, model replacement,
or scientific request retry occurred.

A post-freeze CPU-only candidate adds strict tokenizer-native call-name
recognition and exact role classification. It passes all 32 preserved Gemma
runtime rows, but this is engineering evidence for a successor protocol and
cannot reinterpret or repair consumed v1.1. The next admissible step is a new
frozen two-family qualification whose non-scientific preflight imports the CUDA
extension before any task request and whose production mask is tested against
actual tokenizer-native response tokens.
