"""Execute the exact v3 child-launch gate without a model, server, request, or GPU."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import deployment_entrypoint as gate

STAGE_ROOT = Path(__file__).resolve().parent


def replay(mode: str, output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rist-v3-replay-") as raw_root:
        root = Path(raw_root)
        source = root / "source"
        model = root / "models/snapshots" / ("c" * 40)
        source.mkdir(parents=True)
        model.mkdir(parents=True)
        manifest = source / "manifest.json"
        manifest.write_text('{"protocol":"RIST-C0-v3.0-REPLAY"}\n', encoding="utf-8")
        extension = root / "extension.so"
        extension.write_bytes(b"outcome-free-replay-extension")
        interpreter = Path(sys.executable).resolve(strict=True)
        live = {
            "commit": "a" * 40,
            "gpu_uuid": "GPU-outcome-free-replay",
            "interpreter_sha256": gate._file_sha256(interpreter),
        }

        def file_sha(path: Path) -> str:
            if path.resolve() == interpreter:
                return str(live["interpreter_sha256"])
            return gate._file_sha256(path)

        probes = gate.Probes(
            git_head=lambda _path: str(live["commit"]),
            model_revision=lambda path: path.resolve().name,
            gpu_uuid=lambda: str(live["gpu_uuid"]),
            file_sha256=file_sha,
            interpreter_version=gate._interpreter_version,
        )
        inputs = gate.LiveInputs(source, model, extension, interpreter)
        observed = gate._observed(inputs, manifest.read_bytes(), probes)
        authority = {"protocol": gate.AUTHORITY_PROTOCOL, **observed}
        marker = root / "child.marker"
        receipt = gate.build_receipt(
            inputs=inputs,
            manifest_path=manifest,
            launcher_command=(
                str(interpreter), str(STAGE_ROOT / "fake_server.py"),
                "--marker", str(marker),
            ),
            authority=authority,
            authorized_authority_sha256=gate._bytes_sha256(gate._canonical(authority)),
            probes=probes,
        )
        authorized_receipt_sha = gate._bytes_sha256(gate.receipt_artifact_bytes(receipt))
        if mode == "wrong_worktree":
            live["commit"] = "b" * 40
        elif mode == "wrong_interpreter":
            live["interpreter_sha256"] = "d" * 64
        ledger = root / "ledger.jsonl"
        returncode = gate.launch_receipt(
            receipt,
            inputs=inputs,
            ledger_path=ledger,
            authorized_receipt_sha256=authorized_receipt_sha,
            launcher=gate._run_supervised,
            probes=probes,
        )
        events = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        result = {
            "protocol": "RIST-C0-v3.0-EXACT-DEPLOYMENT-REPLAY-v1",
            "mode": mode,
            "returncode": returncode,
            "marker_created": marker.is_file(),
            "launch_attempt_count": sum(row.get("event") == "launch_attempt" for row in events),
            "decision": next(
                row.get("decision") for row in events
                if row.get("event") == "prelaunch_decision"
            ),
            "model_accessed": False,
            "request_sent": False,
            "gpu_used": False,
            "training_performed": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("success", "wrong_worktree", "wrong_interpreter"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.mode, args.output)
    expected = args.mode == "success"
    passed = (
        result["marker_created"] is expected
        and result["launch_attempt_count"] == (1 if expected else 0)
        and result["decision"] == ("ACCEPT" if expected else "REJECT")
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
