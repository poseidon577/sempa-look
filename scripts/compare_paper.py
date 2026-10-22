"""
Write the paper-vs-ours comparison table.

    python scripts/compare_paper.py [--results results/stage4] [--out docs/reproduction_table.md]
"""

import argparse
from pathlib import Path

from sempalab.compare import compare, load_reference, load_summaries, to_markdown, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/stage4")
    parser.add_argument("--out", default="docs/reproduction_table.md")
    args = parser.parse_args(argv)

    results = Path(args.results)

    rows = compare(load_reference(results / "paper_reference.yaml"), load_summaries(results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(rows))
    write_csv(rows, results / "comparison.csv")

    print(out.read_text())


if __name__ == "__main__":
    main()
