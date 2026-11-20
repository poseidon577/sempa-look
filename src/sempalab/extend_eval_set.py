"""
Grow a frozen evaluation set without re-running the messages it has.

    python -m sempalab.extend_eval_set --config configs/channel/wikihow_k48.yaml --extra 30 --seed 1042

Draws `extra` new K-token messages with another seed, drops any that
already appear in the original set, and writes them as a separate frozen
set <name>_K<K>_n<extra>_seed<seed>.txt. An experiment config with
data.seed = <seed> and data.num_messages = <extra> then runs only the new
messages; `merge_runs` combines the two result files, offsetting the new
message ids so they stay distinct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sempalab.channel_experiment import load_config, read_csv, summarize, write_csv
from sempalab.datasets import load_coco_captions, load_wikihow_sentences, select_captions


def extension_texts(pool: list[str], K: int, original: list[str], extra: int, seed: int) -> list[str]:
    """
    `extra` messages drawn with `seed` that are not in `original`.
    """

    taken = set(original)
    draw = len(original) + extra

    while True:
        candidates = [t for t in select_captions(pool, K, draw, seed) if t not in taken]

        if len(candidates) >= extra:
            return candidates[:extra]

        draw += extra


def set_path(config: dict, num: int, seed: int) -> Path:
    data = config["data"]
    name = "wikihow" + (f"_a{data['max_articles']}" if data["max_articles"] else "") if data["source"] == "wikihow" else "coco_" + Path(data["path"]).stem.replace("captions_", "")
    return Path(data["eval_set_dir"]) / f"{name}_K{config['K']}_n{num}_seed{seed}.txt"


def merge_runs(original_rows: list[dict], extension_rows: list[dict], offset: int = 1000) -> list[dict]:
    if any(int(r["message_id"]) >= offset for r in original_rows):
        raise ValueError("original message ids reach the offset")

    # Keep only algorithms both runs have, so every paired comparison
    # covers the same messages.
    common = {r["algorithm"] for r in original_rows} & {r["algorithm"] for r in extension_rows} if extension_rows and "algorithm" in extension_rows[0] else None
    merged = [r for r in original_rows if common is None or r["algorithm"] in common]

    for row in extension_rows:
        if common is not None and row["algorithm"] not in common:
            continue
        new = dict(row)
        new["message_id"] = int(row["message_id"]) + offset
        merged.append(new)

    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make")
    make.add_argument("--config", required=True)
    make.add_argument("--extra", type=int, required=True)
    make.add_argument("--seed", type=int, required=True)

    merge = sub.add_parser("merge")
    merge.add_argument("--original", required=True, help="results dir of the original run")
    merge.add_argument("--extension", required=True, help="results dir of the extension run")
    merge.add_argument("--out", required=True)
    merge.add_argument("--file", default="results.csv.gz")

    args = parser.parse_args(argv)

    if args.command == "make":
        config = load_config(args.config)
        data = config["data"]
        original = set_path(config, data["num_messages"], data["seed"]).read_text().splitlines()
        pool = load_wikihow_sentences(data["path"], data["max_articles"]) if data["source"] == "wikihow" else load_coco_captions(data["path"])
        texts = extension_texts(pool, config["K"], original, args.extra, args.seed)
        out = set_path(config, args.extra, args.seed)
        out.write_text("\n".join(texts) + "\n")
        print(f"wrote {len(texts)} new messages to {out}")
    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rows = merge_runs(read_csv(Path(args.original) / args.file), read_csv(Path(args.extension) / args.file))
        write_csv(out / args.file, rows)
        summary_name = "summary.csv" if args.file == "results.csv.gz" else args.file.replace("rescored_", "summary_").replace(".csv.gz", ".csv")
        write_csv(out / summary_name, summarize(rows))
        print(f"merged {len(rows)} rows into {out}")


if __name__ == "__main__":
    main()
