"""
Paper-vs-ours comparison for the Stage 4 reproduction.

    python scripts/compare_paper.py

Reads results/stage4/paper_reference.yaml and every
results/stage4/<experiment>/summary.csv, and writes the comparison as
a Markdown table (docs/reproduction_table.md) plus a CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from sempalab.viz import read_summary


def load_reference(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)["entries"]


def _matches(row: dict, spec: dict) -> bool:
    if row["algorithm"] != spec["algorithm"]:
        return False

    for column, value in spec.get("where", {}).items():
        if float(row[column]) != float(value):
            return False

    return True


def lookup(spec: dict, summaries: dict[str, list[dict]]) -> float | None:
    """
    The summary value a spec points at, or None if that run is missing.
    """

    rows = summaries.get(spec["experiment"])

    if rows is None:
        return None

    hits = [row for row in rows if _matches(row, spec)]

    if not hits:
        return None

    if len(hits) > 1:
        raise ValueError(f"Ambiguous spec {spec}: {len(hits)} rows match")

    value = hits[0].get(f"mean_{spec['metric']}")

    return None if value in (None, "") else float(value)


def ours_value(entry: dict, summaries: dict[str, list[dict]]) -> float | None:
    ours = entry["ours"]

    if "value" in ours:
        return lookup(ours["value"], summaries)

    a = lookup(ours["gain"]["a"], summaries)
    b = lookup(ours["gain"]["b"], summaries)

    if a is None or b is None or b == 0:
        return None

    return (a / b - 1) * 100


def compare(entries: list[dict], summaries: dict[str, list[dict]]) -> list[dict]:
    rows = []

    for entry in entries:
        ours = ours_value(entry, summaries)
        paper = float(entry["paper"])

        if ours is None:
            diff = None
        elif entry.get("unit") == "%" or paper == 0:
            diff = ours - paper  # percentage points / absolute
        else:
            diff = (ours - paper) / abs(paper) * 100

        rows.append(
            {
                "id": entry["id"],
                "figure": entry["figure"],
                "quantity": entry["quantity"],
                "setting": entry["setting"],
                "paper": paper,
                "ours": ours,
                "unit": entry.get("unit", ""),
                "diff": diff,
                "diff_kind": "pp" if entry.get("unit") == "%" else ("abs" if paper == 0 else "%"),
                "source": entry["source"],
            }
        )

    return rows


def load_summaries(results_dir: str | Path) -> dict[str, list[dict]]:
    return {
        path.parent.name: read_summary(path)
        for path in sorted(Path(results_dir).glob("*/summary.csv"))
    }


def _fmt(value, unit: str) -> str:
    if value is None:
        return "not run"

    if unit == "%":
        return f"{value:+.2f}%"

    if abs(value) >= 10 or float(value).is_integer():
        return f"{value:,.0f}"

    return f"{value:.4f}"


def _fmt_diff(row: dict) -> str:
    if row["diff"] is None:
        return ""

    if row["diff_kind"] == "pp":
        return f"{row['diff']:+.2f} pp"

    if row["diff_kind"] == "abs":
        return f"{row['diff']:+.0f}"

    return f"{row['diff']:+.1f}%"


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Figure | Quantity | Setting | Paper | Ours | Difference | Paper value from |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| {figure} | {quantity} | {setting} | {paper} | {ours} | {diff} | {source} |".format(
                figure=row["figure"],
                quantity=row["quantity"],
                setting=row["setting"],
                paper=_fmt(row["paper"], row["unit"]),
                ours=_fmt(row["ours"], row["unit"]),
                diff=_fmt_diff(row),
                source="text/table" if row["source"] == "text" else "read off plot",
            )
        )

    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: str | Path) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
