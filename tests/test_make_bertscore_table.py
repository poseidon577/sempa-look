import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("mbt", Path(__file__).parents[1] / "scripts" / "make_bertscore_table.py")
mbt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbt)


def write_summary(path, value):
    fields = ["M", "channel", "parameter", "p", "algorithm"]
    for ref, _ in mbt.REFS:
        fields += [f"gain_vs_{ref}", f"gain_vs_{ref}_ci_low", f"gain_vs_{ref}_ci_high"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for channel, parameter, _ in mbt.CHANNELS:
            for p in ("0.3", "0.5"):
                row = {"M": "2", "channel": channel, "parameter": parameter, "p": p, "algorithm": "casa"}
                for ref, _ in mbt.REFS:
                    row.update({f"gain_vs_{ref}": value, f"gain_vs_{ref}_ci_low": value - 0.01, f"gain_vs_{ref}_ci_high": value + 0.01})
                w.writerow(row)


def test_tables(tmp_path):
    write_summary(tmp_path / "summary.csv", 0.02)
    write_summary(tmp_path / "summary_bertscore.csv", 0.005)
    out = tmp_path / "t.md"
    mbt.main(["--results", str(tmp_path), "--out", str(out)])
    text = out.read_text()
    assert "+0.0200 [+0.0100, +0.0300]" in text
    assert "## N = 6, p = 0.5" in text

    rows = mbt.latex_rows(mbt.load(tmp_path / "summary.csv"), mbt.load(tmp_path / "summary_bertscore.csv"))
    assert len(rows) == len(mbt.CHANNELS)
    assert "$+0.020$" in rows[0]
    assert r"$+0.005^\dagger$" in rows[0]  # CI [-0.005, 0.015] covers zero
