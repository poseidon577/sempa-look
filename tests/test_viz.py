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
                    "mean_wts": 0.09 * snr,
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
