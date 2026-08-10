"""Generate lightweight SVG figures from EVIDENCE_SUMMARY.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "low": "#d73027",
    "ambiguous": "#fee08b",
    "high": "#1a9850",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def heatmap(rows: list[dict[str, str]], output: Path) -> None:
    lineages = ["RIST-C0-v3.1", "RIST-C0-v4.0"]
    families = ["qwen3", "gemma4"]
    cells = [f"c{i:02d}" for i in range(8)]
    lookup = {
        (row["lineage"], row["family"], row["cell"]): row
        for row in rows
    }
    cell_w = 70
    cell_h = 34
    left = 150
    top = 70
    width = left + len(cells) * cell_w + 30
    height = top + len(lineages) * len(families) * cell_h + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="18" font-weight="700">Reward-resolution labels by lineage, model, and cell</text>',
    ]
    for idx, cell in enumerate(cells):
        x = left + idx * cell_w
        parts.append(f'<text x="{x + 18}" y="{top - 14}" font-family="Arial" font-size="12">{cell}</text>')
    row_index = 0
    for lineage in lineages:
        for family in families:
            y = top + row_index * cell_h
            parts.append(
                f'<text x="20" y="{y + 22}" font-family="Arial" font-size="12">{_svg_escape(lineage)} / {_svg_escape(family)}</text>'
            )
            for idx, cell in enumerate(cells):
                row = lookup[(lineage, family, cell)]
                label = row["resolution_label"]
                rate = float(row["strict_success_rate"])
                x = left + idx * cell_w
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{COLORS[label]}" stroke="#333" stroke-width="0.5"/>'
                )
                parts.append(
                    f'<text x="{x + 10}" y="{y + 20}" font-family="Arial" font-size="11">{label[0].upper()} {rate:.2f}</text>'
                )
            row_index += 1
    legend_y = height - 42
    for idx, label in enumerate(("low", "ambiguous", "high")):
        x = left + idx * 120
        parts.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" fill="{COLORS[label]}" stroke="#333" stroke-width="0.5"/>')
        parts.append(f'<text x="{x + 25}" y="{legend_y + 14}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def pipeline(output: Path) -> None:
    width = 980
    height = 190
    boxes = [
        ("Freeze pool", "splits, seeds, thresholds"),
        ("Collect calibration", "zero retry, raw responses"),
        ("Measure resolution", "mixed groups, collapse rates"),
        ("Gate training", "GO only with common contrast"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" font-family="Arial" font-size="18" font-weight="700">Pre-training reward-resolution diagnostic pipeline</text>',
    ]
    x = 40
    y = 70
    box_w = 190
    box_h = 70
    for idx, (title, subtitle) in enumerate(boxes):
        bx = x + idx * 235
        parts.append(f'<rect x="{bx}" y="{y}" width="{box_w}" height="{box_h}" rx="10" fill="#f7f7f7" stroke="#333"/>')
        parts.append(f'<text x="{bx + 16}" y="{y + 30}" font-family="Arial" font-size="15" font-weight="700">{title}</text>')
        parts.append(f'<text x="{bx + 16}" y="{y + 52}" font-family="Arial" font-size="12">{subtitle}</text>')
        if idx < len(boxes) - 1:
            ax = bx + box_w + 10
            parts.append(f'<line x1="{ax}" y1="{y + 35}" x2="{ax + 28}" y2="{y + 35}" stroke="#333" stroke-width="2"/>')
            parts.append(f'<polygon points="{ax + 28},{y + 35} {ax + 18},{y + 29} {ax + 18},{y + 41}" fill="#333"/>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.summary)
    heatmap(rows, args.output_dir / "reward_resolution_heatmap.svg")
    pipeline(args.output_dir / "diagnostic_pipeline.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
