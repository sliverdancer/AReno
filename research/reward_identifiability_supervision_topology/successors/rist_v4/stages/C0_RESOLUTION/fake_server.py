"""Outcome-free child used only by the exact C0 v4 deployment replay."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    args.marker.write_text("exact-bound-interpreter-launch\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
