"""
Re-score stored channel-experiment partitions with BERTScore.

    python -m sempalab.bertscore_rescore --config configs/channel/coco_k12.yaml

Every other number in docs/channel_aware.md uses a sentence-embedding
cosine (MPNet, MiniLM, BGE-small) as phi(F, W). Here phi(F, W) is the
BERTScore F1 between the text of the surviving tokens F (in order) and
the full message W, a token-alignment metric with no sentence pooling.
Partitions and channel laws are unchanged; ATS is recomputed exactly
under each row's law Q.

phi of the empty set stays EMPTY_SIMILARITY (0). That term enters
every algorithm's ATS with the same weight Q(all lost), so paired
differences do not depend on it.

Outputs, next to the experiment's results:
    rescored_bertscore.csv.gz, summary_bertscore.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from sempalab.channel_aware import channel_ats
from sempalab.channel_experiment import LawBook, load_config, load_messages, read_csv, summarize, write_csv
from sempalab.evaluation import from_mask
from sempalab.metrics import EMPTY_SIMILARITY
from sempalab.tokenization import render


DEFAULT_ALGORITHMS = (
    "random_pa",
    "sempa_look",
    "swap_ls+order",
    "interleave_swap_ls+order",
    "casa_iid",
    "casa",
    "full_search_iid",
    "full_search",
)


class BertScorer:
    """
    Thin wrapper over bert_score.BERTScorer returning F1 per pair.
    """

    def __init__(self, model_type: str = "roberta-large", rescale: bool = True, batch_size: int = 64, device: str = "cpu"):
        from bert_score import BERTScorer

        self.name = model_type + ("-rescaled" if rescale else "")
        self.scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=rescale, batch_size=batch_size, device=device)

    def f1(self, candidates: list[str], references: list[str]) -> np.ndarray:
        _, _, f1 = self.scorer.score(candidates, references)
        return f1.numpy().astype(np.float64)


class BertScoreTable:
    """
    Same interface as evaluation.SimilarityTable (`lookup`), with phi
    given by a pairwise text scorer instead of an embedding cosine.
    """

    def __init__(self, scorer, message, reconstruction: str = "fragment", chunk: int = 512, cache_path: Path | None = None):
        self.scorer = scorer
        self.message = message
        self.reconstruction = reconstruction
        self.chunk = chunk
        self.cache_path = cache_path
        self.reference = render(message.tokens, range(len(message.tokens)), reconstruction)
        self.sims: dict[int, float] = {0: EMPTY_SIMILARITY}
        self.calls = 0

        if cache_path is not None and cache_path.exists():
            self.sims.update({int(k): float(v) for k, v in json.loads(cache_path.read_text()).items()})

    def save(self) -> None:
        """
        Persist the scored subsets, so an interrupted re-score resumes.
        """

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({str(k): v for k, v in self.sims.items()}))
            tmp.replace(self.cache_path)

    def fill(self, masks) -> None:
        missing = [mask for mask in dict.fromkeys(int(m) for m in masks) if mask not in self.sims]

        for start in range(0, len(missing), self.chunk):
            batch = missing[start:start + self.chunk]
            texts = [render(self.message.tokens, from_mask(mask), self.reconstruction) for mask in batch]
            scores = self.scorer.f1(texts, [self.reference] * len(texts))
            self.calls += len(batch)

            for mask, score in zip(batch, scores):
                self.sims[mask] = float(score)

    def lookup(self, masks) -> np.ndarray:
        masks = np.asarray(masks, dtype=np.int64)
        self.fill(np.unique(masks).tolist())
        flat = np.fromiter((self.sims[int(m)] for m in masks.ravel()), dtype=np.float64, count=masks.size)
        return flat.reshape(masks.shape)


def rescore_bertscore(config: dict, scorer, algorithms=DEFAULT_ALGORITHMS, M=None, results_name: str = "results.csv.gz", tag: str = "bertscore") -> list[dict]:
    out_dir = Path(config["results_dir"]) / config["experiment"]
    cache_dir = out_dir / f"_cache_{tag}_{getattr(scorer, 'name', 'scorer')}"
    rows = [
        row for row in read_csv(out_dir / results_name)
        if (not algorithms or row["algorithm"] in algorithms) and (M is None or int(row["M"]) in M)
    ]

    laws = LawBook(config)
    messages = {message.message_id: message for message in load_messages(config)}
    reconstruction = config.get("reconstruction", "fragment")

    # Group rows by message so each message's table is built once.
    order = sorted(range(len(rows)), key=lambda i: int(rows[i]["message_id"]))
    rescored = [None] * len(rows)
    table = None

    for i in order:
        row = rows[i]
        message_id = int(row["message_id"])

        if table is None or table.message.message_id != message_id:
            if table is not None:
                table.save()
                print(f"[bertscore] message {table.message.message_id}: {table.calls} pairs scored", flush=True)
            table = BertScoreTable(scorer, messages[message_id], reconstruction, cache_path=cache_dir / f"{message_id}.json")

        partition = tuple(tuple(group) for group in json.loads(row["partition"]))
        point = {"channel": row["channel"], "parameter": float(row["parameter"]), "p": float(row["p"])}
        Q = laws.law(point, int(row["N"]), int(row["stride"]))

        new = dict(row)
        new["encoder"] = tag
        new["partition_encoder"] = row["encoder"]
        new["ats"] = channel_ats(table, partition, Q)
        rescored[i] = new

    if table is not None:
        table.save()

    write_csv(out_dir / f"rescored_{tag}.csv.gz", rescored)
    summary = summarize(rescored)
    write_csv(out_dir / f"summary_{tag}.csv", summary)
    shutil.rmtree(cache_dir, ignore_errors=True)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Re-score channel-experiment partitions with BERTScore F1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="roberta-large")
    parser.add_argument("--no-rescale", action="store_true")
    parser.add_argument("--M", type=int, nargs="*", help="only these subpacket sizes")
    parser.add_argument("--algorithms", nargs="*", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--results-dir")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    overrides = {"results_dir": args.results_dir} if args.results_dir else None
    config = load_config(args.config, overrides)
    scorer = BertScorer(args.model, rescale=not args.no_rescale, batch_size=args.batch_size)

    summary = rescore_bertscore(config, scorer, args.algorithms, args.M)

    for row in summary:
        if row["algorithm"] in {"casa", "sempa_look", "full_search"}:
            print(f"M={row['M']} {row['channel']}({row['parameter']}) p={float(row['p']):.2f} {row['algorithm']:>12} ATS_BS={row['mean_ats']:.4f}")


if __name__ == "__main__":
    main()
