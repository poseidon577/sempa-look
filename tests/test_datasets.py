import json

import pytest

from sempalab.datasets import (
    coco_eval_set,
    load_coco_captions,
    normalize_caption,
    select_captions,
)


def test_normalize_caption():
    assert normalize_caption("A man's T-shirt, on a bed.") == "a man's t-shirt on a bed"
    assert normalize_caption("  Two   dogs -- playing!! ") == "two dogs playing"
    assert normalize_caption("3 cats.") == "3 cats"


def write_annotations(path, captions):
    data = {
        "annotations": [
            {"id": 100 - i, "image_id": i, "caption": caption}
            for i, caption in enumerate(captions)
        ]
    }
    path.write_text(json.dumps(data))


def test_load_orders_by_annotation_id(tmp_path):
    path = tmp_path / "captions_val2017.json"
    write_annotations(path, ["first", "second", "third"])

    # ids are 100, 99, 98 -> sorted ascending reverses the list
    assert load_coco_captions(path) == ["third", "second", "first"]


CAPTIONS = [
    "A cat on a red couch.",          # 6 words
    "A dog runs in the green park.",  # 7
    "a cat on a red couch",           # duplicate after normalization
    "Two birds sit on a wire.",       # 6
    "The man rides a horse home.",    # 6
    "A boat on the lake.",            # 5
]


def test_select_filters_by_length_and_dedupes():
    chosen = select_captions(CAPTIONS, K=6, num_messages=3, seed=0)

    assert sorted(chosen) == sorted(
        ["a cat on a red couch", "two birds sit on a wire", "the man rides a horse home"]
    )
    assert all(len(text.split()) == 6 for text in chosen)


def test_select_is_seeded():
    a = select_captions(CAPTIONS, K=6, num_messages=2, seed=1)
    b = select_captions(CAPTIONS, K=6, num_messages=2, seed=1)

    assert a == b


def test_select_raises_when_pool_too_small():
    with pytest.raises(ValueError):
        select_captions(CAPTIONS, K=6, num_messages=4, seed=0)


def test_eval_set_is_frozen_on_disk(tmp_path):
    path = tmp_path / "captions_val2017.json"
    write_annotations(path, CAPTIONS)

    messages = coco_eval_set(path, K=6, num_messages=2, seed=3, cache_dir=tmp_path)

    cache_file = tmp_path / "coco_val2017_K6_n2_seed3.txt"
    assert cache_file.exists()

    # Changing the source no longer changes the frozen set.
    write_annotations(path, ["x y z w v u"] * 5)

    again = coco_eval_set(path, K=6, num_messages=2, seed=3, cache_dir=tmp_path)

    assert [m.text for m in again] == [m.text for m in messages]
    assert [m.message_id for m in again] == [0, 1]
    assert all(len(m.tokens) == 6 for m in again)


def test_wikihow_sentences_and_eval_set(tmp_path):
    from sempalab.datasets import eval_set, load_wikihow_sentences

    path = tmp_path / "wikihowAll.csv"
    path.write_text(
        'headline,title,text\n'
        '"h1","How to A","Wash the dog with warm water. Dry the dog carefully!\nBrush its fur every day."\n'
        '"h2","How to B","Open the window wide; let the fresh air in now."\n'
    )

    sentences = load_wikihow_sentences(path)
    assert sentences == [
        "Wash the dog with warm water.",
        "Dry the dog carefully!",
        "Brush its fur every day.",
        "Open the window wide;",
        "let the fresh air in now.",
    ]

    assert load_wikihow_sentences(path, max_articles=1) == sentences[:3]

    messages = eval_set("wikihow", path, K=6, num_messages=2, seed=0, cache_dir=tmp_path)
    assert sorted(m.text for m in messages) == ["let the fresh air in now", "wash the dog with warm water"]
    assert (tmp_path / "wikihow_K6_n2_seed0.txt").exists()
