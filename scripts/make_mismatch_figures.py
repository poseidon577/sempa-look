"""
Figure and tables for CASA with a mismatched or estimated channel law.

    python scripts/make_mismatch_figures.py [--results results/channel]

Reads results/channel/coco_k12_mismatch/summary.csv and writes
results/channel/figures/coco_k12_mismatch_estimated.png and
docs/channel_mismatch_tables.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRUTHS = [("gilbert", 0.3), ("gilbert", 0.6), ("gilbert", 0.9), ("rayleigh", 0.9), ("rayleigh", 0.99)]
TRUTH_LABEL = {"gilbert": "Gilbert-Elliott, rho={:g}", "rayleigh": "Rayleigh fading, alpha={:g}"}

DESIGNS = [
    ("casa@matched", "True law (reference)"),
    ("casa@iid", "i.i.d. at true p (memory-blind)"),
    ("casa@gilbert(0.3)", "Gilbert, rho=0.3"),
    ("casa@gilbert(0.6)", "Gilbert, rho=0.6"),
    ("casa@gilbert(0.9)", "Gilbert, rho=0.9"),
    ("casa@rayleigh(0.9)", "Rayleigh, alpha=0.9"),
    ("casa@p_x0.5", "True family, p x 0.5"),
    ("casa@p_x1.5", "True family, p x 1.5"),
    ("casa@est_gilbert(T=100)", "Gilbert fitted to 100 packets"),
    ("casa@est_gilbert(T=300)", "Gilbert fitted to 300 packets"),
    ("casa@est_gilbert(T=1000)", "Gilbert fitted to 1,000 packets"),
    ("casa@est_gilbert(T=10000)", "Gilbert fitted to 10,000 packets"),
    ("casa@est_empirical(T=100)", "Empirical law, 100 packets"),
    ("casa@est_empirical(T=1000)", "Empirical law, 1,000 packets"),
    ("casa@est_empirical(T=10000)", "Empirical law, 10,000 packets"),
    ("swap_ls+order@matched", "Swap LS + slot order, true law"),
    ("sempa_look", "SemPA-Look"),
]

PACKETS = [100, 300, 1000, 10000]


def read(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def pick(rows, M, channel, parameter, p, algorithm):
    for r in rows:
        if r["M"] == M and r["channel"] == channel and r["parameter"] == parameter and r["p"] == p and r["algorithm"] == algorithm:
            return r
    return None


def share_table(rows, M, ps) -> list[str]:
    out = ["| CASA designed with | " + " | ".join(TRUTH_LABEL[c].format(v) for c, v in TRUTHS) + " |",
           "|---|" + "---:|" * len(TRUTHS)]
    for algorithm, label in DESIGNS:
        cells = []
        for channel, parameter in TRUTHS:
            values = [pick(rows, M, channel, parameter, p, algorithm) for p in ps]
            values = [v["share_of_oracle_gain"] for v in values if v and v.get("share_of_oracle_gain") not in (None, "")]
            cells.append(f"{100 * sum(values) / len(values):.0f}%" if values else "")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def gain_table(rows, M, p, reference) -> list[str]:
    out = ["| CASA designed with | " + " | ".join(TRUTH_LABEL[c].format(v) for c, v in TRUTHS) + " |",
           "|---|" + "---|" * len(TRUTHS)]
    for algorithm, label in DESIGNS:
        if algorithm == reference:
            continue
        cells = []
        for channel, parameter in TRUTHS:
            r = pick(rows, M, channel, parameter, p, algorithm)
            key = f"gain_vs_{reference}"
            cells.append(f"{r[key]:+.4f} [{r[key + '_ci_low']:+.4f}, {r[key + '_ci_high']:+.4f}]" if r and key in r and r[key] != "" else "")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def figure(rows, M, ps, path: Path) -> None:
    fig, axes = plt.subplots(1, len(TRUTHS), figsize=(15, 3.4), sharey=True)

    for ax, (channel, parameter) in zip(axes, TRUTHS):
        def mean_share(algorithm):
            values = [pick(rows, M, channel, parameter, p, algorithm) for p in ps]
            return sum(v["share_of_oracle_gain"] for v in values) / len(values)

        ax.axhline(mean_share("casa@matched"), color="#eb6834", linewidth=1.2, linestyle="--", label="CASA, true law")
        ax.axhline(mean_share("swap_ls+order@matched"), color="#eda100", linewidth=1.2, linestyle=":", label="Swap LS + slot order, true law")
        ax.axhline(mean_share("casa@iid"), color="#8a8984", linewidth=1.2, linestyle="-.", label="CASA, i.i.d. law")

        gilbert = [mean_share(f"casa@est_gilbert(T={t})") for t in PACKETS]
        ax.plot(PACKETS, gilbert, color="#eb6834", marker="o", linewidth=2, markersize=6, label="CASA, Gilbert fitted to T packets")

        empirical_t = [100, 1000, 10000]
        empirical = [mean_share(f"casa@est_empirical(T={t})") for t in empirical_t]
        ax.plot(empirical_t, empirical, color="#6250d6", marker="^", linewidth=2, markersize=7, label="CASA, empirical law from T packets")

        ax.set_xscale("log")
        ax.set_title(TRUTH_LABEL[channel].format(parameter), fontsize=10)
        ax.set_xlabel("observed packets T")
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("share of oracle gain")
    axes[0].set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"CASA with an estimated channel law (COCO K=12, N={int(12 // M)}, mean over p = " + ", ".join(f"{p:g}" for p in ps) + ")", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/channel")
    parser.add_argument("--tables", default="docs/channel_mismatch_tables.md")
    args = parser.parse_args(argv)

    results = Path(args.results)
    rows = read(results / "coco_k12_mismatch" / "summary.csv")
    ps = sorted({r["p"] for r in rows})

    figure(rows, 2.0, [p for p in ps if p >= 0.3], results / "figures" / "coco_k12_mismatch_estimated.png")

    lines = [
        "# CASA with a mismatched or estimated channel law: full tables",
        "",
        "Generated by `scripts/make_mismatch_figures.py` from `results/channel/coco_k12_mismatch/summary.csv`. "
        "COCO K = 12, 100 messages, MPNet. CASA is built with the listed law and every partition is scored by exact ATS under the true law (columns). "
        "Share = (ATS − Random PA) / (channel oracle − Random PA). Gains are paired means with bootstrap 95% CIs.",
        "",
    ]
    for M in (2.0, 3.0):
        N = int(12 // M)
        lines += [f"## Share of the oracle's gain, N = {N}, mean over p = " + ", ".join(f"{p:g}" for p in ps), "", *share_table(rows, M, ps), ""]
    for M in (2.0, 3.0):
        N = int(12 // M)
        for p in ps:
            lines += [f"## ATS difference to CASA with the true law, N = {N}, p = {p:g}", "", *gain_table(rows, M, p, "casa@matched"), ""]
    for M in (2.0, 3.0):
        N = int(12 // M)
        for p in ps:
            lines += [f"## ATS difference to swap LS + slot order with the true law, N = {N}, p = {p:g}", "", *gain_table(rows, M, p, "swap_ls+order@matched"), ""]

    Path(args.tables).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.tables} and the figure")


if __name__ == "__main__":
    main()
