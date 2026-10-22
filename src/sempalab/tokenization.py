"""
Word- and subword-based tokenization of messages, and the rule that
turns a received subset of tokens back into text.

Word tokens are the whitespace-split words of the normalized message
(docs/assumptions.md, A2).

Subword tokens come from the WordPiece tokenizer of the MPNet encoder
(sentence-transformers/all-mpnet-base-v2). The paper names BPE and
WordPiece without saying which one it used (Sec. II-A); using the
encoder's own vocabulary is our assumption. A piece that continues a
word carries the "##" prefix.

Reconstruction at RX (`render`), for a received index set:

    fragment  Received pieces are kept in order. A continuation piece
              is glued to the previous piece only when that piece
              (index i - 1) was received too; otherwise it starts a new
              fragment with "##" stripped. A word that lost a piece is
              therefore only partly recovered ("skate ##board ##er"
              with "##board" lost becomes "skate er").
    strict    A word survives only when all of its pieces are received;
              partial words are dropped.

For word tokens (no "##" pieces) both rules reduce to joining the
received tokens with single spaces, the reconstruction used for every
word-based result.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence


TOKENIZATIONS = ("word", "subword")

RECONSTRUCTIONS = ("fragment", "strict")

SUBWORD_TOKENIZER = "sentence-transformers/all-mpnet-base-v2"

CONTINUATION = "##"


@lru_cache(maxsize=1)
def subword_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SUBWORD_TOKENIZER)


def tokenize(text: str, tokenization: str = "word") -> tuple[str, ...]:
    if tokenization == "word":
        return tuple(text.split())

    if tokenization == "subword":
        return tuple(subword_tokenizer().tokenize(text))

    raise ValueError(f"Unknown tokenization: {tokenization}")


def word_spans(tokens: Sequence[str]) -> list[tuple[int, ...]]:
    """
    Group token indices into words: a piece starting with "##"
    belongs to the word of the piece before it.
    """

    spans: list[list[int]] = []

    for i, token in enumerate(tokens):

        if token.startswith(CONTINUATION) and spans:
            spans[-1].append(i)
        else:
            spans.append([i])

    return [tuple(span) for span in spans]


def render(
    tokens: Sequence[str],
    indices: Iterable[int],
    reconstruction: str = "fragment",
) -> str:
    """
    Text reconstructed from the tokens at `indices` (in token order).
    """

    received = sorted(set(indices))

    if reconstruction == "strict":
        keep = set(received)
        received = [
            i
            for span in word_spans(tokens)
            if all(j in keep for j in span)
            for i in span
        ]

    elif reconstruction != "fragment":
        raise ValueError(f"Unknown reconstruction: {reconstruction}")

    words: list[str] = []
    previous = None

    for i in received:

        token = tokens[i]

        if token.startswith(CONTINUATION):
            piece = token[len(CONTINUATION):]

            if words and previous == i - 1:
                words[-1] += piece
            else:
                words.append(piece)
        else:
            words.append(token)

        previous = i

    return " ".join(words)
