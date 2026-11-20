import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("mmf", Path(__file__).resolve().parents[1] / "scripts" / "make_mismatch_figures.py")
mmf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mmf)


def rows():
    out = []
    for channel, parameter in mmf.TRUTHS:
        for p in (0.3, 0.5):
            for algorithm, _ in mmf.DESIGNS:
                out.append({"M": 2.0, "channel": channel, "parameter": parameter, "p": p, "algorithm": algorithm,
                            "share_of_oracle_gain": 0.5 if algorithm == "casa@iid" else 0.9,
                            "gain_vs_casa@matched": -0.01, "gain_vs_casa@matched_ci_low": -0.02, "gain_vs_casa@matched_ci_high": 0.0})
    return out


def test_share_table_averages_over_p():
    table = mmf.share_table(rows(), 2.0, [0.3, 0.5])
    assert len(table) == 2 + len(mmf.DESIGNS)
    iid = next(line for line in table if line.startswith("| i.i.d."))
    assert iid.count("50%") == len(mmf.TRUTHS)


def test_gain_table_skips_reference():
    table = mmf.gain_table(rows(), 2.0, 0.3, "casa@matched")
    assert not any(line.startswith("| True law") for line in table)
    assert "-0.0100 [-0.0200, +0.0000]" in table[2]


def test_figure_writes_png(tmp_path):
    path = tmp_path / "f.png"
    mmf.figure(rows(), 2.0, [0.3, 0.5], path)
    assert path.stat().st_size > 0
