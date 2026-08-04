"""Signal-safe successor launcher for the frozen RIST C0 v2.3 deployment gate.

The original deployment entrypoint remains unchanged. This successor reuses
its receipt and live-identity validation, but forwards termination only to the
direct server child so the server can close its rollout workers itself.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path
from typing import Sequence

from deployment_entrypoint import LiveInputs, launch_receipt


def _run_supervised(command: Sequence[str]) -> int:
    child = subprocess.Popen(list(command))
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def forward_to_server(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in previous_handlers:
            signal.signal(signum, forward_to_server)
        return int(child.wait())
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authorized-receipt-sha256", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    return launch_receipt(
        receipt,
        inputs=LiveInputs(
            args.control_root,
            args.runtime_root,
            args.model_path,
            args.extension,
        ),
        ledger_path=args.ledger,
        authorized_receipt_sha256=args.authorized_receipt_sha256,
        receipt_bytes=receipt_bytes,
        launcher=_run_supervised,
    )


if __name__ == "__main__":
    raise SystemExit(main())
