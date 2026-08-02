def byte_identical(
    observed_bytes,
    regenerated_bytes,
    observed_fragment,
    regenerated_fragment,
):
    """Treat generated JSONL bytes as the sole reproducibility truth."""

    return observed_bytes == regenerated_bytes
