"""
Regenerate every Stage 4 figure from results/stage4/*/summary.csv.

    python scripts/make_figures.py [--results results/stage4]

Experiments without a sweep are plotted against SNR. Swept experiments
get one figure per SNR, plotted against the swept parameter.
"""

import argparse
from pathlib import Path

from sempalab.viz import plot_vs_x, read_summary


AXIS_LABEL = {"M": "Subpacket size M", "k": "Lookahead width k", "P": "Candidates P", "K": "Message length K"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/stage4")
    args = parser.parse_args(argv)

    results = Path(args.results)
    figures = results / "figures"

    for summary_path in sorted(results.glob("*/summary.csv")):

        rows = read_summary(summary_path)
        experiment = summary_path.parent.name
        encoder = rows[0]["encoder"]
        param = rows[0]["sweep_param"]

        if not param:
            out = figures / f"{experiment}.png"
            plot_vs_x(rows, "snr_db", "SNR (dB)", title=f"{experiment} ({encoder})", out_path=out)
            print(out)
            continue

        for snr in sorted({r["snr_db"] for r in rows}):
            subset = [r for r in rows if r["snr_db"] == snr]
            out = figures / f"{experiment}_snr{snr}.png"
            plot_vs_x(
                subset,
                "sweep_value",
                AXIS_LABEL.get(param, param),
                title=f"{experiment} at {snr} dB ({encoder})",
                out_path=out,
            )
            print(out)


if __name__ == "__main__":
    main()
