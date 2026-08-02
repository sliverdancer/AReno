# Parent held-out retirement notice

The file
`../rist_v2/stages/D2/data/heldout.jsonl` is permanently retired from all RIST
scientific evidence.

On 2026-08-02, during CPU-only source auditing, a broad command equivalent to
`find ... | xargs rg -n "calibrat|resolution|stratum"` covered the parent D2
data directory. The command did not print held-out content, but the search
process could open and scan the file. Under the fail-closed protocol, lack of a
printed match does not restore sealing.

Consequences:

- do not evaluate, summarize, hash-compare, or otherwise reuse the parent D2
  held-out as confirmatory evidence;
- do not reinterpret the event as a negative scientific result;
- D4 uses fresh split names and nonces, commits confirmatory bytes by hash, and
  materializes confirmatory rows in memory only after a one-shot ledger exists.
