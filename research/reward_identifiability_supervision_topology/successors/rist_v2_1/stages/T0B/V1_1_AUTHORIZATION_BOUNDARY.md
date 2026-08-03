# T0b v1.1 new-instance acquisition boundary

Status: `FROZEN_AWAITING_EXPLICIT_WEIGHT_DOWNLOAD_AUTHORIZATION`

The newly rented instance is reachable and provides one idle RTX 4090D, a
30 GB system disk, and a 200 GB mounted data disk. It contains neither frozen
checkpoint. T0b v1.0 forbids checkpoint downloads, so execution cannot reuse
that protocol unchanged.

T0b v1.1 freezes a minimal 11,798,002,411-byte payload at the exact T0a
revisions. The acquisition lock records every required path, byte count, and
Git-blob SHA-1 or LFS SHA-256. The downloader refuses to run while
`download_permitted=false`, refuses existing output directories, downloads no
unlisted files, materializes no symlinks, and admits a snapshot only after the
complete file set and digests pass.

Changing `download_permitted` requires explicit user authorization and a new
commit before remote execution. It does not authorize training, held-out data,
BFCL content, model substitution, or any revision other than the two locked
commits.
