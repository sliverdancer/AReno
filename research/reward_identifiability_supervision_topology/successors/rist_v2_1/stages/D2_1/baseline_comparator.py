"""Archived failure mechanism reconstructed inside the successor harness."""


def byte_identical(
    observed_bytes,
    regenerated_bytes,
    observed_fragment,
    regenerated_fragment,
):
    return (
        observed_bytes == regenerated_bytes
        and observed_fragment == regenerated_fragment
    )
