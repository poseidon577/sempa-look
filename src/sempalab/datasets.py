"""
MS-COCO caption loading for the Stage 4 reproduction.

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

import json
import re
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


def to_messages(texts: list[str]) -> list[Message]:
    return [
        Message(message_id=i, text=text, tokens=tuple(text.split()))
        for i, text in enumerate(texts)
    ]


def coco_eval_set(
    annotations_path: str | Path,
    K: int,
    num_messages: int,
    seed: int,
    cache_dir: str | Path | None = None,
) -> list[Message]:
    """
    Build (or reload) a frozen evaluation set.

    When `cache_dir` is given the selected captions are written to
    <cache_dir>/coco_<split>_K<K>_n<num>_seed<seed>.txt and reused
    afterwards, so every experiment sees the same messages.
    """

    annotations_path = Path(annotations_path)

    cache_file = None

    if cache_dir is not None:
        split = annotations_path.stem.replace("captions_", "")
        cache_file = (
            Path(cache_dir)
            / f"coco_{split}_K{K}_n{num_messages}_seed{seed}.txt"
        )

        if cache_file.exists():
            return to_messages(cache_file.read_text().splitlines())

    texts = select_captions(
        load_coco_captions(annotations_path),
        K,
        num_messages,
        seed,
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(texts) + "\n")

    return to_messages(texts)
