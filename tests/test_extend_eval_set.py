import pytest

from sempalab.extend_eval_set import extension_texts, merge_runs


POOL = [f"w{i} a b c" for i in range(50)]


def test_extension_excludes_original_and_is_deterministic():
    original = POOL[:10]
    a = extension_texts(POOL, 4, original, 15, seed=7)
    b = extension_texts(POOL, 4, original, 15, seed=7)
    assert a == b
    assert len(a) == 15 and len(set(a)) == 15
    assert not set(a) & set(original)
    assert all(len(t.split()) == 4 for t in a)


def test_extension_grows_draw_when_overlap_is_large():
    original = POOL[:45]
    assert len(extension_texts(POOL, 4, original, 5, seed=0)) == 5


def test_merge_offsets_ids():
    original = [{"message_id": "0", "ats": "0.5"}, {"message_id": "29", "ats": "0.6"}]
    extension = [{"message_id": "0", "ats": "0.7"}]
    merged = merge_runs(original, extension)
    assert [int(r["message_id"]) for r in merged] == [0, 29, 1000]
    with pytest.raises(ValueError):
        merge_runs([{"message_id": "1000"}], extension)


def test_merge_keeps_common_algorithms_only():
    original = [{"message_id": "0", "algorithm": "casa"}, {"message_id": "0", "algorithm": "interleave_swap_ls"}]
    extension = [{"message_id": "0", "algorithm": "casa"}]
    merged = merge_runs(original, extension)
    assert {r["algorithm"] for r in merged} == {"casa"}
    assert len(merged) == 2
