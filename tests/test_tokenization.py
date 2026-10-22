import json

import pytest

from sempalab.datasets import eval_set, select_captions
from sempalab.repro import load_repro_config, validate_repro_config
from sempalab.tokenization import render, tokenize, word_spans


# "a skateboarder rides" as MPNet WordPiece pieces.
PIECES = ("a", "skate", "##board", "##er", "rides")


def test_word_render_is_space_join():
    tokens = ("a", "dog", "runs", "fast")

    for rule in ("fragment", "strict"):
        assert render(tokens, (3, 0, 2), rule) == "a runs fast"
        assert render(tokens, (), rule) == ""


def test_word_spans_group_continuations():
    assert word_spans(PIECES) == [(0,), (1, 2, 3), (4,)]

    # A leading continuation piece starts its own word.
    assert word_spans(("##ing", "x")) == [(0,), (1,)]


def test_fragment_rule():
    assert render(PIECES, range(5)) == "a skateboarder rides"

    # Head lost: the remaining contiguous pieces form a fragment.
    assert render(PIECES, (0, 2, 3, 4)) == "a boarder rides"

    # Middle piece lost: the pieces on either side are not glued.
    assert render(PIECES, (0, 1, 3, 4)) == "a skate er rides"

    # Only a tail piece survives.
    assert render(PIECES, (3,)) == "er"


def test_strict_rule_drops_partial_words():
    assert render(PIECES, range(5), "strict") == "a skateboarder rides"
    assert render(PIECES, (0, 1, 3, 4), "strict") == "a rides"
    assert render(PIECES, (1, 2, 3), "strict") == "skateboarder"


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        render(PIECES, (0,), "nope")

    with pytest.raises(ValueError):
        tokenize("a cat", "chars")


def test_subword_tokenize_round_trips():
    text = "a skateboarder doing a kickflip on a half-pipe"
    pieces = tokenize(text, "subword")

    assert any(piece.startswith("##") for piece in pieces)
    assert len(pieces) > len(text.split())

    # Full reception reproduces the text up to spaces around "-".
    assert render(pieces, range(len(pieces))) == (
        "a skateboarder doing a kickflip on a half - pipe"
    )


def test_select_captions_counts_subword_pieces():
    captions = [
        "a skateboarder rides",    # 3 words, 5 pieces
        "a dog runs in a park",    # 6 words, 6 pieces
        "the cat sat on a mat",    # 6 words, 6 pieces
    ]

    assert select_captions(captions, 5, 1, seed=0, tokenization="subword") == [
        "a skateboarder rides"
    ]
    assert select_captions(captions, 3, 1, seed=0) == ["a skateboarder rides"]
    assert len(select_captions(captions, 6, 2, seed=0, tokenization="subword")) == 2


def test_subword_eval_set_cached_separately(tmp_path):
    path = tmp_path / "captions_val2017.json"
    captions = ["a skateboarder rides", "a dog runs in a park"]
    path.write_text(json.dumps({
        "annotations": [
            {"id": i, "image_id": i, "caption": c} for i, c in enumerate(captions)
        ]
    }))

    messages = eval_set("coco", path, 5, 1, 0, cache_dir=tmp_path,
                        tokenization="subword")

    assert messages[0].tokens == PIECES
    assert (tmp_path / "coco_val2017_subword_K5_n1_seed0.txt").exists()

    # Reloading from the cache tokenizes the same way.
    again = eval_set("coco", path, 5, 1, 0, cache_dir=tmp_path,
                     tokenization="subword")
    assert again == messages


def test_repro_config_validates_tokenization():
    config = load_repro_config("configs/repro/coco_k8_subword.yaml")
    assert config["tokenization"] == "subword"

    config["reconstruction"] = "nope"
    with pytest.raises(ValueError):
        validate_repro_config(config)
