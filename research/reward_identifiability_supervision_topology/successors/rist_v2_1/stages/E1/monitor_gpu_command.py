"""Run one command while recording exact NVIDIA GPU memory identity and peak."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

QUERY = "uuid,name,memory.total,memory.used,driver_version"


def _query() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, name, total, used, driver = [value.strip() for value in line.split(",", 4)]
        rows.append(
            {
                "gpu_uuid": uuid,
                "gpu_name": name,
                "total_memory_mib": float(total),
                "used_memory_mib": float(used),
                "driver_version": driver,
            }
        )
    return rows


def monitor(command: list[str], gpu_uuid: str, interval: float) -> tuple[int, dict]:
    matches = [row for row in _query() if row["gpu_uuid"] == gpu_uuid]
    if len(matches) != 1:
        raise ValueError("requested E1 GPU UUID is not uniquely visible")
    identity = matches[0]
    process = subprocess.Popen(command)
    samples = []
    started = time.monotonic()
    while process.poll() is None:
        current = [row for row in _query() if row["gpu_uuid"] == gpu_uuid]
        if len(current) != 1:
            process.terminate()
            raise RuntimeError("E1 GPU disappeared during monitored command")
        samples.append(
            {
                "elapsed_seconds": time.monotonic() - started,
                "used_memory_mib": current[0]["used_memory_mib"],
            }
        )
        time.sleep(interval)
    final = [row for row in _query() if row["gpu_uuid"] == gpu_uuid][0]
    samples.append(
        {
            "elapsed_seconds": time.monotonic() - started,
            "used_memory_mib": final["used_memory_mib"],
        }
    )
    peak = max(row["used_memory_mib"] for row in samples)
    return process.returncode, {
        "protocol": "RIST-E1-GPU-MONITOR-v1",
        **identity,
        "sample_count": len(samples),
        "peak_memory_mib": peak,
        "peak_memory_gib": peak / 1024.0,
        "command_exit_code": process.returncode,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("monitored command is required")
    code, result = monitor(command, args.gpu_uuid, args.interval_seconds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
