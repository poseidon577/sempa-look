"""
Figures and tables for the channel-aware aggregation results.

    python scripts/make_channel_figures.py [--results results/channel]

Reads results/channel/<experiment>/summary*.csv and writes
results/channel/figures/*.png and docs/channel_aware_tables.md.
Experiments that have not been run are skipped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Fixed categorical order (one colour per algorithm, never re-cycled).
STYLE = {
    "random_pa": ("Random PA", "#8a8984", "o", ":"),
    "interleave": ("Interleaving", "#e87ba4", "v", ":"),
    "sempa_look": ("SemPA-Look", "#2a78d6", "s", "-"),
    "swap_ls": ("Swap LS (RSS)", "#eda100", "D", "-"),
    "sempa_look+order": ("SemPA-Look + slot order", "#2a78d6", "s", "--"),
    "swap_ls+order": ("Swap LS + slot order", "#eda100", "D", "--"),
    "casa_iid": ("CASA, i.i.d. model", "#6250d6", "^", "--"),
    "casa": ("CASA", "#eb6834", "o", "-"),
    "full_search_iid": ("Full Search (i.i.d.)", "#1baf7a", "P", "-"),
    "full_search": ("Full Search (channel oracle)", "#0b0b0b", "*", "--"),
}

CHANNEL_LABEL = {
    "iid": "i.i.d.",
    "gilbert": "Gilbert-Elliott, rho={:g}",
    "rayleigh": "Rayleigh fading, alpha={:g}",
}

ENCODER_LABEL = {"mpnet": "MPNet (optimised)", "minilm": "MiniLM", "bge-small": "BGE-small"}


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass

    return rows


def select(rows, **where):
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def channel_label(channel, parameter):
    return CHANNEL_LABEL[channel].format(parameter)


def style_axis(ax):
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bottom_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=8, frameon=False)


def line(ax, rows, x, y, name, err=None):
    label, color, marker, dash = STYLE[name]
    rows = sorted(rows, key=lambda r: r[x])
    xs = [r[x] for r in rows]
    ys = [r[y] for r in rows]
    ax.plot(xs, ys, color=color, marker=marker, linestyle=dash, linewidth=2, markersize=6, label=label)
    if err:
        low = [r[err[0]] for r in rows]
        high = [r[err[1]] for r in rows]
        ax.fill_between(xs, low, high, color=color, alpha=0.12, linewidth=0)


def gain_vs_p(summary, M, channels, algorithms, out, title):
    fig, axes = plt.subplots(1, len(channels), figsize=(4.3 * len(channels), 3.6), sharey=True)

    for ax, (channel, parameter) in zip(axes, channels):
        rows = select(summary, M=M, channel=channel, parameter=parameter, stride=1)
        for name in algorithms:
            line(ax, select(rows, algorithm=name), "p", "gain_vs_random_pa",
                 name, err=("gain_vs_random_pa_ci_low", "gain_vs_random_pa_ci_high"))
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(channel_label(channel, parameter), fontsize=10)
        ax.set_xlabel("average packet error rate p")
        style_axis(ax)

    axes[0].set_ylabel("ATS gain over Random PA")
    bottom_legend(fig, axes[0])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def memory_sweep(summary, p, algorithms, out, title):
    Ms = sorted({r["M"] for r in summary}, reverse=True)
    fig, axes = plt.subplots(2, len(Ms), figsize=(4.0 * len(Ms), 6.8), sharey="row", squeeze=False)

    for col, M in enumerate(Ms):
        N = int(12 // M)
        for row, (channel, symbol) in enumerate((("gilbert", "rho"), ("rayleigh", "alpha"))):
            ax = axes[row][col]
            rows = select(summary, M=M, p=p, stride=1)
            for name in algorithms:
                points = [r for r in rows if r["algorithm"] == name and r["channel"] in ("iid", channel)]
                line(ax, points, "parameter", "gain_vs_random_pa", name)
            ax.set_title(f"N={N}, {channel}", fontsize=10)
            ax.set_xlabel(f"channel memory {symbol} (0 = i.i.d.)")
            style_axis(ax)
        axes[0][col].set_ylabel("ATS gain over Random PA")
        axes[1][col].set_ylabel("ATS gain over Random PA")

    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")
    bottom_legend(fig, axes[0][0])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def stride_sweep(summary, p, channels, algorithms, out, title):
    Ms = sorted({r["M"] for r in summary}, reverse=True)
    fig, axes = plt.subplots(len(Ms), len(channels), figsize=(4.3 * len(channels), 3.4 * len(Ms)), squeeze=False)

    for i, M in enumerate(Ms):
        for j, (channel, parameter) in enumerate(channels):
            ax = axes[i][j]
            rows = select(summary, M=M, p=p, channel=channel, parameter=parameter)
            iid = {r["algorithm"]: r["mean_ats"] for r in select(summary, M=M, p=p, channel="iid")}
            for name in algorithms:
                line(ax, select(rows, algorithm=name), "stride", "mean_ats", name)
            if "full_search_iid" in iid:
                ax.axhline(iid["full_search_iid"], color="#1baf7a", linewidth=1, linestyle=":")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
            ax.set_title(f"N={int(12 // M)}, {channel_label(channel, parameter)}", fontsize=10)
            ax.set_xlabel("interleaver depth (packets between subpackets)")
            ax.set_ylabel("ATS")
            style_axis(ax)

    bottom_legend(fig, axes[0][0])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def rescore_figure(summaries, M, channels, out, title):
    fig, axes = plt.subplots(1, len(channels), figsize=(4.3 * len(channels), 3.6), sharey=True)
    colors = {"mpnet": "#2a78d6", "minilm": "#eb6834", "bge-small": "#1baf7a"}
    markers = {"mpnet": "s", "minilm": "o", "bge-small": "D"}

    for ax, (channel, parameter) in zip(axes, channels):
        for encoder, summary in summaries.items():
            rows = sorted(select(summary, M=M, channel=channel, parameter=parameter, stride=1, algorithm="casa"),
                          key=lambda r: r["p"])
            xs = [r["p"] for r in rows]
            ax.plot(xs, [r["gain_vs_sempa_look"] for r in rows], color=colors[encoder], marker=markers[encoder],
                    linewidth=2, label=ENCODER_LABEL[encoder])
            ax.fill_between(xs, [r["gain_vs_sempa_look_ci_low"] for r in rows],
                            [r["gain_vs_sempa_look_ci_high"] for r in rows], color=colors[encoder], alpha=0.12, linewidth=0)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(channel_label(channel, parameter), fontsize=10)
        ax.set_xlabel("average packet error rate p")
        style_axis(ax)

    axes[0].set_ylabel("CASA ATS gain over SemPA-Look")
    bottom_legend(fig, axes[0])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


# -------------------------------------------------------------
# Tables
# -------------------------------------------------------------


def fmt_gain(row, reference):
    key = f"gain_vs_{reference}"
    if key not in row or row[key] == "":
        return ""
    return f"{row[key]:+.4f} [{row[key + '_ci_low']:+.4f}, {row[key + '_ci_high']:+.4f}]"


def main_table(summary, M, p_values, channels, algorithms) -> str:
    lines = []
    for channel, parameter in channels:
        lines.append(f"\n**{channel_label(channel, parameter)}**, N = {int(12 // M)} (M = {int(M)})\n")
        lines.append("| Algorithm | calls | " + " | ".join(f"p = {p:g}" for p in p_values) + " |")
        lines.append("|---|---:|" + "---:|" * len(p_values))
        rows = select(summary, M=M, channel=channel, parameter=parameter, stride=1)
        for name in algorithms:
            by_p = {r["p"]: r for r in rows if r["algorithm"] == name}
            if not by_p:
                continue
            calls = sum(r["mean_logical_calls"] for r in by_p.values()) / len(by_p)
            cells = [f"{by_p[p]['mean_ats']:.4f}" if p in by_p else "" for p in p_values]
            lines.append(f"| {STYLE[name][0]} | {calls:,.0f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gain_table(summary, M, p_values, channels, reference) -> str:
    lines = [
        "| Channel | " + " | ".join(f"p = {p:g}" for p in p_values) + " |",
        "|---|" + "---|" * len(p_values),
    ]
    for channel, parameter in channels:
        cells = []
        for p in p_values:
            rows = select(summary, M=M, channel=channel, parameter=parameter, stride=1, p=p, algorithm="casa")
            cells.append(fmt_gain(rows[0], reference) if rows else "")
        lines.append(f"| {channel_label(channel, parameter)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headroom_table(summary, p_values, channels, algorithms) -> str:
    """
    Share of the oracle's gain over Random PA that each algorithm
    captures, (ATS - Random) / (Oracle - Random), averaged over p.
    """

    Ms = sorted({r["M"] for r in summary}, reverse=True)
    lines = [
        "| Channel | " + " | ".join(f"{STYLE[a][0]}, N={int(12 // M)}" for M in Ms for a in algorithms) + " |",
        "|---|" + "---:|" * (len(Ms) * len(algorithms)),
    ]
    for channel, parameter in channels:
        cells = []
        for M in Ms:
            for name in algorithms:
                shares = []
                for p in p_values:
                    by = {r["algorithm"]: r["mean_ats"] for r in select(summary, M=M, channel=channel, parameter=parameter, stride=1, p=p)}
                    shares.append((by[name] - by["random_pa"]) / (by["full_search"] - by["random_pa"]))
                cells.append(f"{100 * sum(shares) / len(shares):.0f}%")
        lines.append(f"| {channel_label(channel, parameter)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/channel")
    parser.add_argument("--tables", default="docs/channel_aware_tables.md")
    args = parser.parse_args(argv)

    results = Path(args.results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def load(experiment, name="summary.csv"):
        path = results / experiment / name
        return read(path) if path.exists() else None

    tables = ["# Channel-aware aggregation: full tables\n",
              "Generated by `scripts/make_channel_figures.py` from `results/channel/*/summary*.csv`. "
              "ATS is exact under each channel's joint slot law, mean over 100 messages, MPNet unless stated. "
              "Gains are paired means with bootstrap 95% CIs. Calls are mean logical encoder calls per message.\n"]

    main_algos = ["sempa_look", "swap_ls", "swap_ls+order", "casa", "full_search_iid", "full_search"]
    all_algos = list(STYLE)

    coco = load("coco_k12")
    if coco:
        p_values = sorted({r["p"] for r in coco})
        channels = list(dict.fromkeys((r["channel"], r["parameter"]) for r in coco))
        shown = [("iid", 0.0), ("gilbert", 0.6), ("rayleigh", 0.9)]

        for M in sorted({r["M"] for r in coco}, reverse=True):
            N = int(12 // M)
            gain_vs_p(coco, M, shown, main_algos, out / f"coco_k12_N{N}_gain_vs_p.png",
                      f"COCO K=12, N={N}: ATS gain over Random PA (MPNet, 100 messages, 95% CI)")

        memory_sweep(coco, 0.3, main_algos, out / "coco_k12_memory_p0.3.png",
                     "COCO K=12, p=0.3: gain over Random PA as channel memory grows")

        tables.append("\n## COCO K = 12: share of the oracle's gain over Random PA, averaged over p = 0.2 to 0.6\n")
        tables.append(headroom_table(coco, [p for p in p_values if p >= 0.2], channels,
                                     ["sempa_look", "swap_ls+order", "full_search_iid", "casa"]))

        tables.append("\n## COCO K = 12: mean ATS\n")
        for M in sorted({r["M"] for r in coco}, reverse=True):
            tables.append(main_table(coco, M, p_values, channels, all_algos))

        tables.append("\n## COCO K = 12: CASA gain over SemPA-Look\n")
        for M in sorted({r["M"] for r in coco}, reverse=True):
            tables.append(f"\nN = {int(12 // M)}\n")
            tables.append(gain_table(coco, M, p_values, channels, "sempa_look"))

        tables.append("\n## COCO K = 12: CASA gain over Full Search (i.i.d.), the best channel-blind partition\n")
        for M in sorted({r["M"] for r in coco}, reverse=True):
            tables.append(f"\nN = {int(12 // M)}\n")
            tables.append(gain_table(coco, M, p_values, channels, "full_search_iid"))

        rescored = {"mpnet": coco}
        for encoder in ("minilm", "bge-small"):
            other = load("coco_k12", f"summary_{encoder}.csv")
            if other:
                rescored[encoder] = other

        if len(rescored) > 1:
            for M in sorted({r["M"] for r in coco}, reverse=True):
                N = int(12 // M)
                rescore_figure(rescored, M, shown, out / f"coco_k12_N{N}_rescored.png",
                               f"COCO K=12, N={N}: CASA gain over SemPA-Look, partitions chosen with MPNet")
            tables.append("\n## Re-scored under other encoders (partitions chosen with MPNet)\n")
            for encoder, summary in rescored.items():
                if encoder == "mpnet":
                    continue
                for M in sorted({r["M"] for r in coco}, reverse=True):
                    tables.append(f"\n**{ENCODER_LABEL[encoder]}**, N = {int(12 // M)}: CASA gain over SemPA-Look\n")
                    tables.append(gain_table(summary, M, p_values, channels, "sempa_look"))

    stride = load("coco_k12_stride")
    if stride:
        strong = [("gilbert", 0.9), ("rayleigh", 0.99)]
        for p in (0.3, 0.5):
            stride_sweep(stride, p, strong, ["sempa_look", "swap_ls", "swap_ls+order", "casa", "full_search"],
                         out / f"coco_k12_stride_p{p}.png",
                         f"COCO K=12, p={p}: ATS vs packet interleaver depth (dotted: i.i.d. optimum)")
        tables.append("\n## Interleaver depth (COCO K = 12)\n")
        tables.append("| N | Channel | p | Algorithm | depth 1 | depth 2 | depth 4 | depth 8 |")
        tables.append("|---|---|---:|---|---:|---:|---:|---:|")
        for M in sorted({r["M"] for r in stride}, reverse=True):
            for channel, parameter in strong:
                for p in (0.1, 0.3, 0.5):
                    for name in ["sempa_look", "swap_ls+order", "casa", "full_search"]:
                        rows = {r["stride"]: r["mean_ats"] for r in select(stride, M=M, channel=channel, parameter=parameter, p=p, algorithm=name)}
                        if rows:
                            tables.append(f"| {int(12 // M)} | {channel_label(channel, parameter)} | {p:g} | {STYLE[name][0]} | "
                                          + " | ".join(f"{rows.get(d, float('nan')):.4f}" for d in (1, 2, 4, 8)) + " |")

    matched = load("coco_k12_matched")
    if matched:
        p_values = sorted({r["p"] for r in matched})
        channels = list(dict.fromkeys((r["channel"], r["parameter"]) for r in matched))
        tables.append("\n## Compute-matched ablation (swap LS with 3 restarts, CASA with 158 fitting subsets)\n")
        for M in sorted({r["M"] for r in matched}, reverse=True):
            tables.append(main_table(matched, M, p_values, channels, ["sempa_look", "swap_ls", "swap_ls+order", "casa", "full_search"]))

    wikihow = load("wikihow_k24")
    if wikihow:
        p_values = sorted({r["p"] for r in wikihow})
        channels = list(dict.fromkeys((r["channel"], r["parameter"]) for r in wikihow))
        tables.append("\n## WikiHow K = 24, N = 6 (no oracle)\n")
        for M in sorted({r["M"] for r in wikihow}, reverse=True):
            tables.append(main_table(wikihow, M, p_values, channels,
                                     ["random_pa", "interleave", "sempa_look", "swap_ls", "sempa_look+order", "swap_ls+order", "casa_iid", "casa"]))
        gain_vs_p_k24 = [c for c in [("iid", 0.0), ("gilbert", 0.6), ("rayleigh", 0.9)] if c in channels]
        gain_vs_p(wikihow, 4.0, gain_vs_p_k24, ["sempa_look", "swap_ls", "swap_ls+order", "casa_iid", "casa"],
                  out / "wikihow_k24_N6_gain_vs_p.png", "WikiHow K=24, N=6: ATS gain over Random PA (MPNet, 95% CI)")

    Path(args.tables).write_text("\n".join(tables) + "\n")
    print(args.tables)


if __name__ == "__main__":
    main()
