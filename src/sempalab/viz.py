"""
Figures for the Stage 4 reproduction, drawn from summary.csv files.

    python scripts/make_figures.py

Every algorithm keeps one color and marker in every figure, so the
figures can be read side by side.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# Fixed categorical order: color follows the algorithm, never its rank.
STYLE = {
    "sempa_look": ("SemPA-Look", "#2a78d6", "o"),
    "full_search": ("Full Search", "#0b0b0b", "s"),
    "genetic": ("Genetic", "#1baf7a", "^"),
    "greedy_rss": ("Greedy RSS", "#eb6834", "D"),
    "greedy_tss": ("Greedy TSS", "#eda100", "v"),
    "random_pa": ("Random PA", "#e34948", "x"),
    "no_packetization": ("No packetization", "#6250d6", "P"),
}

METRIC_LABEL = {"ats": "ATS", "wts": "WTS"}

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def read_summary(path: str | Path) -> list[dict]:
    rows = []

    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                try:
                    row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                except (ValueError, AttributeError):
                    pass
            rows.append(row)

    return rows


def _style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)

    ax.tick_params(colors=MUTED)


def _plot_series(ax, rows, x_key, y_key, label_suffix=""):
    algorithms = [name for name in STYLE if any(r["algorithm"] == name for r in rows)]

    for name in algorithms:
        series = sorted((r for r in rows if r["algorithm"] == name), key=lambda r: r[x_key])

        label, color, marker = STYLE[name]

        ax.plot(
            [r[x_key] for r in series],
            [r[y_key] for r in series],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=label + label_suffix,
        )


def plot_vs_x(
    summary: list[dict],
    x_key: str,
    xlabel: str,
    metrics: tuple[str, ...] = ("ats", "wts"),
    title: str = "",
    out_path: str | Path | None = None,
    algorithms: list[str] | None = None,
):
    """
    One panel per metric, one line per algorithm, x = x_key.
    """

    rows = [r for r in summary if algorithms is None or r["algorithm"] in algorithms]

    metrics = tuple(m for m in metrics if f"mean_{m}" in rows[0])

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.9), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        _plot_series(ax, rows, x_key, f"mean_{metric}")
        _style_axes(ax, xlabel, METRIC_LABEL[metric])

    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=TEXT)

    if title:
        fig.suptitle(title, color=TEXT, fontsize=11)

    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return fig
