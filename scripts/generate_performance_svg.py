#!/usr/bin/env python3
"""
Generate a log-log performance comparison SVG from the recorded timings in
docs/performance_log.txt. No external dependencies required.

Output: docs/img/performance_comparison.svg
"""

import math
from pathlib import Path

# Recorded timings (seconds) from docs/performance_log.txt
DATA = {
    "sequential": [
        (10_000, 1.047),
        (100_000, 11.480),
        (1_000_000, 215.044),
    ],
    "multiprocessing": [
        (10_000, 0.377),
        (100_000, 2.937),
        (1_000_000, 60.163),
    ],
    "mpi (4 ranks)": [
        (10_000, 0.001),
        (100_000, 0.004),
        (1_000_000, 0.028),
        (10_000_000, 0.213),
    ],
}

COLORS = {
    "sequential": "#d62728",
    "multiprocessing": "#1f77b4",
    "mpi (4 ranks)": "#2ca02c",
}


def log10(x: float) -> float:
    if x <= 0:
        raise ValueError("Values must be positive for log10 axis")
    return math.log10(x)


def main() -> None:
    # Axis ranges on log-log scale
    x_min = log10(10_000)
    x_max = log10(10_000_000)
    y_min = min(log10(t) for series in DATA.values() for _, t in series)
    y_max = max(log10(t) for series in DATA.values() for _, t in series)
    y_pad = 0.2
    y_min -= y_pad
    y_max += y_pad

    width, height = 900, 560
    margin_left, margin_bottom, margin_top, margin_right = 120, 80, 60, 40

    def scale_x(n: float) -> float:
        return margin_left + (log10(n) - x_min) / (x_max - x_min) * (width - margin_left - margin_right)

    def scale_y(t: float) -> float:
        return height - margin_bottom - (log10(t) - y_min) / (y_max - y_min) * (height - margin_bottom - margin_top)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text { font-family: sans-serif; font-size: 13px; }</style>')

    # Axes
    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, height - margin_bottom
    x_axis = f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="black" stroke-width="1.5" />'
    y_axis = f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="black" stroke-width="1.5" />'
    parts.extend([x_axis, y_axis])

    # X ticks (dataset sizes)
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        x = scale_x(n)
        parts.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + 6}" stroke="black" />')
        label = f"{n:,}"
        parts.append(f'<text x="{x}" y="{y0 + 24}" text-anchor="middle">{label}</text>')

    # Y ticks (times, logarithmic)
    for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        if log10(t) < y_min or log10(t) > y_max:
            continue
        y = scale_y(t)
        parts.append(f'<line x1="{x0 - 6}" y1="{y}" x2="{x0}" y2="{y}" stroke="black" />')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end">{t:g}s</text>')

    # Title and axis labels
    parts.append(f'<text x="{width/2}" y="{margin_top - 20}" text-anchor="middle" font-size="18">Radix Sort Performance (log-log)</text>')
    parts.append(f'<text x="{(x0 + x1)/2}" y="{height - 20}" text-anchor="middle">Input size (n)</text>')
    parts.append(f'<text x="25" y="{(margin_top + y0)/2}" text-anchor="middle" transform="rotate(-90 25 {(margin_top + y0)/2})">Time (seconds, log scale)</text>')

    # Series plots
    for name, series in DATA.items():
        color = COLORS[name]
        coords = [f"{scale_x(n):.2f},{scale_y(t):.2f}" for n, t in series]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}" />')
        for n, t in series:
            parts.append(
                f'<circle cx="{scale_x(n):.2f}" cy="{scale_y(t):.2f}" r="4" fill="{color}" stroke="white" stroke-width="1.5">'
                f'<title>{name}: n={n:,}, t={t:.3f}s</title></circle>'
            )

    # Legend
    legend_x, legend_y = width - margin_right - 200, margin_top + 10
    line_height = 22
    parts.append(f'<rect x="{legend_x - 10}" y="{legend_y - 14}" width="180" height="{len(DATA)*line_height + 10}" fill="#f8f8f8" stroke="#ccc" />')
    for i, (name, color) in enumerate(COLORS.items()):
        y = legend_y + i * line_height
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1.5" />')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 5}" >{name}</text>')

    parts.append("</svg>")

    out_path = Path("docs/img/performance_comparison.svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path} from hardcoded data.")


if __name__ == "__main__":
    main()
