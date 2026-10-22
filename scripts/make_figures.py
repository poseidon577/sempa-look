"""
Regenerate every Stage 4 figure from results/stage4/*/summary.csv.

    python scripts/make_figures.py [--results results/stage4]

Solid lines are our runs; dashed lines with hollow markers are the
paper's curves (results/stage4/paper_curves.yaml, read off the PDF).
Figures whose run is missing are skipped.
"""

import argparse
from pathlib import Path

import yaml

from sempalab.viz import plot_panels, read_summary, wts_by_L


def select(rows, **where):
    return [r for r in rows if all(float(r[k]) == float(v) for k, v in where.items())]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/stage4")
    args = parser.parse_args(argv)

    results = Path(args.results)
    out = results / "figures"
    paper = yaml.safe_load((results / "paper_curves.yaml").read_text())

    runs = {p.parent.name: read_summary(p) for p in results.glob("*/summary.csv")}

    def figure(name, needs, panels, title):
        if not all(n in runs for n in needs):
            print(f"skip {name}: missing {[n for n in needs if n not in runs]}")
            return
        path = out / f"{name}.png"
        plot_panels(panels(), title=title, out_path=path)
        print(path)

    snr_ticks = [2, 4, 6, 8, 10]
    M_ticks = [1, 2, 3, 4, 6, 12]

    figure(
        "fig3_tss_vs_rss",
        ["coco_k12"],
        lambda: [
            {
                "rows": [r for r in select(runs["coco_k12"], M=3) if r["algorithm"] in ("greedy_tss", "greedy_rss", "random_pa")],
                "x_key": "snr_db", "y_key": "mean_ats", "xlabel": "SNR (dB)", "ylabel": "ATS",
                "paper": paper["fig3"], "xticks": snr_ticks,
            }
        ],
        "Fig. 3: TSS- vs RSS-based PA (COCO, ours K=12, M=3)",
    )

    figure(
        "fig6_ats_wts",
        ["coco_k8", "coco_k12"],
        lambda: [
            {
                "rows": [r for r in runs["coco_k8"] if r["algorithm"] not in ("greedy_tss", "greedy_rss")],
                "x_key": "snr_db", "y_key": "mean_ats", "xlabel": "SNR (dB)", "ylabel": "ATS",
                "paper": paper["fig6a"], "xticks": snr_ticks, "title": "(a) ATS, K=8, M=4",
            },
            {
                "rows": wts_by_L([r for r in select(runs["coco_k12"], M=2, snr_db=6)
                                  if r["algorithm"] in ("full_search", "genetic", "sempa_look", "random_pa")]),
                "x_key": "L", "y_key": "mean_wts", "xlabel": "Lost subpackets L", "ylabel": "WTS",
                "paper": paper["fig6c"], "xticks": [1, 2, 3, 4, 5], "title": "(c) WTS, K=12, M=2 (N=6), 6 dB",
            },
        ],
        "Fig. 6: ATS and WTS of PA methods (MS-COCO, MPNet)",
    )

    main_algos = ("full_search", "genetic", "sempa_look", "random_pa")

    figure(
        "fig8_ats_vs_M",
        ["coco_k12"],
        lambda: [
            {
                "rows": [r for r in select(runs["coco_k12"], snr_db=7) if r["algorithm"] in main_algos],
                "x_key": "M", "y_key": "mean_ats", "xlabel": "Subpacket size M", "ylabel": "ATS",
                "paper": paper["fig8"], "xticks": M_ticks,
            }
        ],
        "Fig. 7a / 8: ATS vs M (COCO, K=12, 7 dB)",
    )

    figure(
        "fig9_ats_vs_M_4_8dB",
        ["coco_k12"],
        lambda: [
            {
                "rows": [r for r in select(runs["coco_k12"], snr_db=snr) if r["algorithm"] in main_algos],
                "x_key": "M", "y_key": "mean_ats", "xlabel": "Subpacket size M", "ylabel": "ATS",
                "paper": paper[key], "xticks": M_ticks, "title": f"{snr} dB",
            }
            for snr, key in ((4, "fig9a"), (8, "fig9b"))
        ],
        "Fig. 9: ATS vs M under extreme SNR (COCO, K=12)",
    )

    figure(
        "fig10_k_and_G",
        ["wikihow_k24_m4", "wikihow_ga_g"],
        lambda: [
            {
                "rows": select(runs["wikihow_k24_m4"], snr_db=7),
                "x_key": "k", "y_key": "mean_ats", "xlabel": "Lookahead width k (k_max = 5)", "ylabel": "ATS",
                "paper": paper["fig10a"], "xticks": [0, 1, 2, 3, 4, 5], "title": "(a) ATS vs k",
            },
            {
                "rows": runs["wikihow_ga_g"],
                "x_key": "G", "y_key": "mean_ats", "xlabel": "GA generations G", "ylabel": "ATS",
                "paper": paper["fig10b"], "title": "(b) GA vs G (30 messages)",
            },
        ],
        "Fig. 10: SemPA-Look vs k and GA vs G (WikiHow, K=24, M=4, 7 dB)",
    )

    figure(
        "fig11_P",
        ["wikihow_p"],
        lambda: [
            {
                "rows": runs["wikihow_p"], "x_key": "P", "y_key": "mean_ats", "xlabel": "P", "ylabel": "ATS",
                "paper": paper["fig11"], "title": "(a) ATS",
            },
            {
                "rows": runs["wikihow_p"], "x_key": "P", "y_key": "mean_logical_calls", "xlabel": "P",
                "ylabel": "Encoder calls per message", "title": "(b) Complexity",
            },
        ],
        "Fig. 11: P sweep (WikiHow, K=24, M=12, 7 dB, 30 messages)",
    )

    def length_rows(metric):
        rows = list(runs["wikihow_len"])
        rows += runs.get("wikihow_len_long", [])
        return rows

    figure(
        "fig12_length",
        ["wikihow_len"],
        lambda: [
            {
                "rows": length_rows("ats"), "x_key": "K", "y_key": "mean_ats", "xlabel": "Sentence length K",
                "ylabel": "ATS", "paper": paper["fig12a"], "title": "(a) ATS",
            },
            {
                "rows": length_rows("wts"), "x_key": "K", "y_key": "mean_wts_L1", "xlabel": "Sentence length K",
                "ylabel": "WTS (L=1)", "paper": paper["fig12b"], "title": "(b) WTS",
            },
        ],
        "Fig. 12: ATS and WTS vs K with M=5 (WikiHow, 7 dB)",
    )

    figure(
        "fig13_fixed_N",
        ["wikihow_fixed_n2", "wikihow_fixed_n4"],
        lambda: [
            {
                "rows": runs[f"wikihow_fixed_n{n}"], "x_key": "K", "y_key": "mean_ats", "xlabel": "Sentence length K",
                "ylabel": "ATS", "paper": paper[f"fig13_n{n}"], "title": f"N = {n}",
            }
            for n in (2, 4)
        ],
        "Fig. 13: ATS vs K with N fixed (WikiHow, 7 dB)",
    )


if __name__ == "__main__":
    main()
