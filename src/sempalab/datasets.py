"""
Message datasets for the Stage 4 reproduction: MS-COCO captions and
WikiHow sentences.

Only the caption annotations are needed (no images):

    https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
        annotations/captions_train2017.json
        annotations/captions_val2017.json

Normalization (assumption, see docs/assumptions.md): lowercase, every
character that is not a letter, digit, apostrophe or hyphen becomes a
space, then whitespace word tokenization. A caption is a candidate
message of length K when it has exactly K words after normalization.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from sempalab.types import Message


COCO_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/"
    "annotations/annotations_trainval2017.zip"
)

_NON_WORD = re.compile(r"[^a-z0-9'\-]+")


def normalize_caption(caption: str) -> str:
    """
    Lowercase, drop punctuation, collapse whitespace.
    """

    words = _NON_WORD.sub(" ", caption.lower()).split()

    # Stray hyphens / apostrophes left as their own word carry no meaning.
    words = [word for word in words if word.strip("'-")]

    return " ".join(words)


def load_coco_captions(path: str | Path) -> list[str]:
    """
    Read a captions_*.json file and return its captions in
    annotation-id order (stable across machines).
    """

    with Path(path).open() as handle:
        data = json.load(handle)

    annotations = sorted(data["annotations"], key=lambda a: a["id"])

    return [annotation["caption"] for annotation in annotations]


def select_captions(
    captions: list[str],
    K: int,
    num_messages: int,
    seed: int,
) -> list[str]:
    """
    Normalize, keep unique captions with exactly K words, and draw
    `num_messages` of them uniformly at random without replacement.

    Selection depends only on (captions, K, num_messages, seed).
    """

    if K <= 0 or num_messages <= 0:
        raise ValueError("K and num_messages must be positive")

    pool = []
    seen = set()

    for caption in captions:

        text = normalize_caption(caption)

        if len(text.split()) != K or text in seen:
            continue

        seen.add(text)
        pool.append(text)

    if len(pool) < num_messages:
        raise ValueError(
            f"Only {len(pool)} unique captions with K={K} words, "
            f"need {num_messages}"
        )

    rng = np.random.default_rng(seed)

    chosen = rng.choice(len(pool), size=num_messages, replace=False)

    return [pool[i] for i in sorted(chosen)]


WIKIHOW_CSV_URL = (
    "https://huggingface.co/datasets/Lajvi/wikihowAll/resolve/main/wikihowAll.csv"
)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|\n+")


def load_wikihow_sentences(path: str | Path, max_articles: int | None = None) -> list[str]:
    """
    Sentences from the `text` column of wikihowAll.csv (Koupaee and
    Wang, 2018), in file order. Assumption (docs/assumptions.md): a
    WikiHow message is one sentence of an article body, split at
    . ! ? ; or a line break.
    """

    csv.field_size_limit(sys.maxsize)

    sentences = []

    with Path(path).open(newline="", encoding="utf-8") as handle:
        for count, row in enumerate(csv.DictReader(handle)):
            if max_articles is not None and count >= max_articles:
                break

            sentences.extend(
                part for part in _SENTENCE_BREAK.split(row.get("text") or "") if part.strip()
            )

    return sentences


def to_messages(texts: list[str]) -> list[Message]:
    return [
        Message(message_id=i, text=text, tokens=tuple(text.split()))
        for i, text in enumerate(texts)
    ]


def eval_set(
    source: str,
    path: str | Path,
    K: int,
    num_messages: int,
    seed: int,
    cache_dir: str | Path | None = None,
    max_articles: int | None = None,
) -> list[Message]:
    """
    Build (or reload) a frozen evaluation set of K-word messages.

    source is "coco" (path = captions_*.json) or "wikihow"
    (path = wikihowAll.csv, first `max_articles` articles).

    When `cache_dir` is given the selected texts are written to
    <cache_dir>/<name>_K<K>_n<num>_seed<seed>.txt and reused afterwards,
    so every experiment sees the same messages.
    """

    path = Path(path)

    if source == "coco":
        name = "coco_" + path.stem.replace("captions_", "")
    elif source == "wikihow":
        name = "wikihow" + (f"_a{max_articles}" if max_articles else "")
    else:
        raise ValueError(f"Unknown source: {source}")

    cache_file = None

    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{name}_K{K}_n{num_messages}_seed{seed}.txt"

        if cache_file.exists():
            return to_messages(cache_file.read_text().splitlines())

    if source == "coco":
        pool = load_coco_captions(path)
    else:
        pool = load_wikihow_sentences(path, max_articles)

    texts = select_captions(pool, K, num_messages, seed)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(texts) + "\n")

    return to_messages(texts)


def coco_eval_set(annotations_path, K, num_messages, seed, cache_dir=None):
    return eval_set("coco", annotations_path, K, num_messages, seed, cache_dir)
