import csv

from sempalab.viz import STYLE, plot_vs_x, read_summary


def write_summary(path):
    rows = []
    for name in ("sempa_look", "random_pa"):
        for snr in (4, 6, 8):
            rows.append(
                {
                    "algorithm": name,
                    "snr_db": snr,
                    "mean_ats": 0.1 * snr,
                    "mean_wts_L1": 0.09 * snr,
                    "mean_logical_calls": 20,
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_read_summary_parses_numbers(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path)

    rows = read_summary(path)

    assert rows[0]["snr_db"] == 4
    assert isinstance(rows[0]["mean_ats"], float)
    assert rows[0]["algorithm"] == "sempa_look"


def test_plot_writes_png(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path)

    out = tmp_path / "fig.png"
    fig = plot_vs_x(read_summary(path), "snr_db", "SNR (dB)", out_path=out)

    assert out.exists() and out.stat().st_size > 1000
    assert len(fig.axes) == 2


def test_every_algorithm_has_a_distinct_color():
    colors = [color for _, color, _ in STYLE.values()]

    assert len(set(colors)) == len(colors)


def test_paper_overlay_and_wts_by_L(tmp_path):
    from sempalab.viz import plot_panels, wts_by_L

    summary = [
        {"algorithm": "sempa_look", "mean_wts_L1": 0.8, "mean_wts_L2": 0.6, "mean_ats": 0.9},
        {"algorithm": "random_pa", "mean_wts_L1": 0.7, "mean_wts_L2": 0.5, "mean_ats": 0.85},
    ]

    rows = wts_by_L(summary)
    assert {(r["algorithm"], r["L"]) for r in rows} == {
        ("sempa_look", 1), ("sempa_look", 2), ("random_pa", 1), ("random_pa", 2)
    }

    out = tmp_path / "wts.png"
    fig = plot_panels(
        [{"rows": rows, "x_key": "L", "y_key": "mean_wts", "xlabel": "L", "ylabel": "WTS",
          "paper": {"x": [1, 2], "sempa_look": [0.82, 0.62], "random_pa": [0.73, 0.5]}}],
        out_path=out,
    )

    assert out.exists()
    # 2 ours + 2 paper lines
    assert len(fig.axes[0].lines) == 4
