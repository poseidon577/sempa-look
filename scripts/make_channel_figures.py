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

from sempalab.delay import matching_depth, span_packets  # noqa: E402


# Fixed categorical order (one colour per algorithm, never re-cycled).
STYLE = {
    "random_pa": ("Random PA", "#8a8984", "o", ":"),
    "interleave": ("Interleaving", "#e87ba4", "v", ":"),
    "sempa_look": ("SemPA-Look", "#2a78d6", "s", "-"),
    "swap_ls": ("Swap LS (RSS)", "#eda100", "D", "-"),
    "sempa_look+order": ("SemPA-Look + slot order", "#2a78d6", "s", "--"),
    "swap_ls+order": ("Swap LS + slot order", "#eda100", "D", "--"),
    "interleave+order": ("Interleaving + slot order", "#e87ba4", "v", "--"),
    "interleave_swap_ls": ("Interleaver-seeded swap LS", "#008f9c", "X", "-"),
    "interleave_swap_ls+order": ("Interleaver-seeded swap LS + order", "#008f9c", "X", "--"),
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


def bottom_legend(fig, ax, ncol=6):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), ncol), fontsize=8, frameon=False)


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
            ticks = sorted({int(r["stride"]) for r in rows})
            ax.set_xticks(ticks, [str(t) for t in ticks])
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


def delay_vs_gain(summary, p, channels, algorithms, out, title):
    """
    ATS gain over Random PA at depth 1 (the paper's layout, no
    interleaving) against the message span (N-1)*D in packets. Reading
    across at a fixed gain gives the delay each method needs.
    """

    Ms = sorted({r["M"] for r in summary}, reverse=True)
    fig, axes = plt.subplots(len(Ms), len(channels), figsize=(4.3 * len(channels), 3.4 * len(Ms)), squeeze=False)

    for i, M in enumerate(Ms):
        N = int(round(summary[0]["K"] / M))
        for j, (channel, parameter) in enumerate(channels):
            ax = axes[i][j]
            rows = select(summary, M=M, p=p, channel=channel, parameter=parameter)
            base = select(rows, algorithm="random_pa", stride=1)[0]["mean_ats"]
            for name in algorithms:
                label, color, marker, dash = STYLE[name]
                points = sorted(select(rows, algorithm=name), key=lambda r: r["stride"])
                ax.plot([span_packets(N, int(r["stride"])) for r in points], [r["mean_ats"] - base for r in points],
                        color=color, marker=marker, linestyle=dash, linewidth=2, markersize=6, label=label)
            iid = select(summary, M=M, p=p, channel="iid", algorithm="full_search_iid")
            if iid:
                ax.axhline(iid[0]["mean_ats"] - base, color="#1baf7a", linewidth=1, linestyle=":")
            ax.set_xscale("log", base=2)
            spans = sorted({span_packets(N, int(r["stride"])) for r in rows})
            ax.set_xticks(spans, [str(s) for s in spans])
            ax.set_title(f"N={N}, {channel_label(channel, parameter)}", fontsize=10)
            ax.set_xlabel("delay: message span (N-1)D, packets")
            if j == 0:
                ax.set_ylabel("ATS gain over Random PA at D=1")
            style_axis(ax)

    bottom_legend(fig, axes[0][0], ncol=3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def depth_ratio_table(summary, channels, p_values, algorithms, reference="casa") -> str:
    """
    For each blind method, how much deeper an interleaver it needs to
    match `reference` at depth D: matching depth / D, interpolated over
    log2 D. '>' marks a method that never matches within the depths run.
    """

    Ms = sorted({r["M"] for r in summary}, reverse=True)
    depths = sorted({int(r["stride"]) for r in summary if r["channel"] != "iid"})
    shown = [d for d in depths if d < depths[-1]]
    lines = [
        "| N | Channel | p | Method | " + " | ".join(f"match CASA at D={d}" for d in shown) + " |",
        "|---|---|---:|---|" + "---:|" * len(shown),
    ]
    for M in Ms:
        N = int(round(summary[0]["K"] / M))
        for channel, parameter in channels:
            for p in p_values:
                rows = select(summary, M=M, p=p, channel=channel, parameter=parameter)
                target = {int(r["stride"]): r["mean_ats"] for r in select(rows, algorithm=reference)}
                for name in algorithms:
                    curve = sorted((int(r["stride"]), r["mean_ats"]) for r in select(rows, algorithm=name))
                    if not curve:
                        continue
                    cells = []
                    for d in shown:
                        need = matching_depth([c[0] for c in curve], [c[1] for c in curve], target[d])
                        cells.append(f"> {depths[-1] / d:.0f}x" if need == float("inf") else f"{need / d:.2f}x")
                    lines.append(f"| {N} | {channel_label(channel, parameter)} | {p:g} | {STYLE[name][0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    return main_table_k(summary, 12, M, p_values, channels, algorithms)


def main_table_k(summary, K, M, p_values, channels, algorithms) -> str:
    lines = []
    for channel, parameter in channels:
        lines.append(f"\n**{channel_label(channel, parameter)}**, N = {int(K // M)} (M = {int(M)})\n")
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

    def present(rows, names):
        return [n for n in names if select(rows, algorithm=n)]

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
                                     [a for a in ["sempa_look", "swap_ls+order", "interleave_swap_ls+order", "full_search_iid", "casa"]
                                      if select(coco, algorithm=a)]))

        tables.append("\n## COCO K = 12: mean ATS\n")
        for M in sorted({r["M"] for r in coco}, reverse=True):
            tables.append(main_table(coco, M, p_values, channels, present(coco, all_algos)))

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
        depths = sorted({int(r["stride"]) for r in stride})
        delay_algos = [a for a in ["sempa_look", "sempa_look+order", "swap_ls+order", "interleave_swap_ls+order", "casa", "full_search"]
                       if select(stride, algorithm=a)]
        for p in (0.3, 0.5):
            stride_sweep(stride, p, strong, [a for a in ["sempa_look", "swap_ls", "swap_ls+order", "interleave_swap_ls+order", "casa", "full_search"]
                                             if select(stride, algorithm=a)],
                         out / f"coco_k12_stride_p{p}.png",
                         f"COCO K=12, p={p}: ATS vs packet interleaver depth (dotted: i.i.d. optimum)")
            delay_vs_gain(stride, p, strong, delay_algos, out / f"coco_k12_delay_p{p}.png",
                          f"COCO K=12, p={p}: gain vs delay, packet interleaving at depth D=" + "/".join(map(str, depths))
                          + " (dotted: i.i.d. optimum)")
        shown_algos = [a for a in ["random_pa", "interleave+order", "sempa_look", "sempa_look+order", "swap_ls+order",
                                   "interleave_swap_ls+order", "casa", "full_search"] if select(stride, algorithm=a)]
        tables.append("\n## Interleaver depth (COCO K = 12)\n")
        tables.append("Mean ATS at packet interleaver depth D; the message span is (N-1)D packets.\n")
        tables.append("| N | Channel | p | Algorithm | " + " | ".join(f"D = {d}" for d in depths) + " |")
        tables.append("|---|---|---:|---|" + "---:|" * len(depths))
        for M in sorted({r["M"] for r in stride}, reverse=True):
            for channel, parameter in strong:
                for p in (0.1, 0.3, 0.5):
                    for name in shown_algos:
                        rows = {r["stride"]: r["mean_ats"] for r in select(stride, M=M, channel=channel, parameter=parameter, p=p, algorithm=name)}
                        if rows:
                            tables.append(f"| {int(12 // M)} | {channel_label(channel, parameter)} | {p:g} | {STYLE[name][0]} | "
                                          + " | ".join(f"{rows.get(d, float('nan')):.4f}" for d in depths) + " |")
        tables.append("\n## Delay a channel-blind method needs to match CASA (COCO K = 12)\n")
        tables.append("Interleaver depth at which the method's ATS reaches CASA's ATS at depth D, divided by D "
                      "(linear in log2 D). Delay scales with depth, so 2.00x means twice CASA's delay for the same quality.\n")
        tables.append(depth_ratio_table(stride, strong, (0.1, 0.3, 0.5),
                                        [a for a in ["sempa_look", "sempa_look+order", "swap_ls+order", "interleave_swap_ls+order"]
                                         if select(stride, algorithm=a)]))

    matched = load("coco_k12_matched")
    if matched:
        p_values = sorted({r["p"] for r in matched})
        channels = list(dict.fromkeys((r["channel"], r["parameter"]) for r in matched))
        tables.append("\n## Compute-matched ablation (swap LS with 3 restarts, CASA with 158 fitting subsets)\n")
        for M in sorted({r["M"] for r in matched}, reverse=True):
            tables.append(main_table(matched, M, p_values, channels, ["sempa_look", "swap_ls", "swap_ls+order", "casa", "full_search"]))

    long_algos = ["random_pa", "interleave", "sempa_look", "swap_ls", "interleave_swap_ls", "sempa_look+order",
                  "swap_ls+order", "interleave_swap_ls+order", "casa_iid", "casa"]
    for K in (24, 48, 60):
        wikihow = load(f"wikihow_k{K}")
        if not wikihow:
            continue
        p_values = sorted({r["p"] for r in wikihow})
        channels = list(dict.fromkeys((r["channel"], r["parameter"]) for r in wikihow))
        messages = int(wikihow[0]["messages"])
        for M in sorted({r["M"] for r in wikihow}, reverse=True):
            N = int(K // M)
            tables.append(f"\n## WikiHow K = {K}, N = {N} ({messages} messages, no oracle)\n")
            tables.append(main_table_k(wikihow, K, M, p_values, channels, [a for a in long_algos if select(wikihow, algorithm=a)]))
            for reference in ("sempa_look", "swap_ls+order", "interleave_swap_ls+order"):
                if select(wikihow, algorithm=reference):
                    tables.append(f"\nCASA gain over {STYLE[reference][0]}, K = {K}\n")
                    tables.append(gain_table(wikihow, M, p_values, channels, reference))
            for encoder in ("minilm", "bge-small"):
                other = load(f"wikihow_k{K}", f"summary_{encoder}.csv")
                if other:
                    tables.append(f"\n**{ENCODER_LABEL[encoder]}** re-score, K = {K}: CASA gain over SemPA-Look\n")
                    tables.append(gain_table(other, M, p_values, channels, "sempa_look"))
            shown = [c for c in [("iid", 0.0), ("gilbert", 0.6), ("rayleigh", 0.9)] if c in channels]
            gain_vs_p(wikihow, M, shown, [a for a in ["sempa_look", "swap_ls+order", "interleave_swap_ls+order", "casa_iid", "casa"]
                                          if select(wikihow, algorithm=a)],
                      out / f"wikihow_k{K}_N{N}_gain_vs_p.png",
                      f"WikiHow K={K}, N={N}: ATS gain over Random PA (MPNet, {messages} messages, 95% CI)")

    Path(args.tables).write_text("\n".join(tables) + "\n")
    print(args.tables)


if __name__ == "__main__":
    main()
