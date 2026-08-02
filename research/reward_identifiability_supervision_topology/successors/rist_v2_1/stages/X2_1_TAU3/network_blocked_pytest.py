"""Run a pytest invocation after denying outbound socket connections."""

from __future__ import annotations

import socket
import sys

import pytest


def _denied(_socket, address):
    raise RuntimeError(
        f"outbound network connection denied during Tau3 qualification: {address}"
    )


def main() -> int:
    socket.socket.connect = _denied
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
