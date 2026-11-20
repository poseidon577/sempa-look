import json

import numpy as np
import pytest
import yaml

from sempalab.bertscore_rescore import BertScoreTable, rescore_bertscore
from sempalab.channel.correlated import GilbertElliott
from sempalab.channel_aware import union_masks
from sempalab.channel_experiment import load_config, read_csv, run
from sempalab.types import Message


TEXTS = [
    "a man rides a red bike near the sea",
    "two dogs play with a ball in green grass",
    "a plate of food with rice and fresh greens",
]


class JaccardScorer:
    """Deterministic stand-in for BERTScore: word-set Jaccard."""

    name = "jaccard"

    def __init__(self):
        self.pairs = 0

    def f1(self, candidates, references):
        self.pairs += len(candidates)
        out = []
        for c, r in zip(candidates, references):
            a, b = set(c.split()), set(r.split())
            out.append(len(a & b) / len(a | b))
        return np.array(out)


def test_table_scores_rendered_subsets_and_caches():
    words = TEXTS[0].split()[:8]
    message = Message(0, " ".join(words), tuple(words))
    scorer = JaccardScorer()
    table = BertScoreTable(scorer, message)

    full = (1 << 8) - 1
    values = table.lookup([0, full, 0b11, 0b11])
    assert values[0] == 0.0  # empty set
    assert values[1] == pytest.approx(1.0)
    # tokens 0,1 = "a man" vs 7 distinct words in the message
    assert values[2] == pytest.approx(2 / len(set(words)))
    assert scorer.pairs == 2  # cached + deduplicated
    table.lookup([full])
    assert scorer.pairs == 2


def test_rescore_keeps_partitions_and_is_exact(tmp_path, fake_encoder):
    eval_dir = tmp_path / "eval_sets"
    eval_dir.mkdir()
    (eval_dir / "coco_val2017_K8_n3_seed0.txt").write_text("\n".join(" ".join(t.split()[:8]) for t in TEXTS))
    raw = {
        "experiment": "bs",
        "data": {"source": "coco", "path": str(tmp_path / "captions_val2017.json"), "num_messages": 3, "seed": 0, "eval_set_dir": str(eval_dir)},
        "results_dir": str(tmp_path / "results"),
        "encoder": "minilm",
        "K": 8,
        "M": [2, 4],
        "p": [0.3],
        "channels": [{"name": "gilbert", "parameter": 0.6}],
        "algorithms": ["random_pa", "casa", "full_search"],
        "r2_samples": 0,
    }
    path = tmp_path / "bs.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    run(config, encoder=fake_encoder())

    rescore_bertscore(config, JaccardScorer(), algorithms=("casa", "full_search"), M=[2])

    original = read_csv(tmp_path / "results" / "bs" / "results.csv.gz")
    rescored = read_csv(tmp_path / "results" / "bs" / "rescored_bertscore.csv.gz")

    kept = [r for r in original if r["algorithm"] in {"casa", "full_search"} and r["M"] == "2"]
    assert sorted(r["partition"] for r in kept) == sorted(r["partition"] for r in rescored)
    assert all(r["encoder"] == "bertscore" and r["partition_encoder"] == "minilm" for r in rescored)

    Q = GilbertElliott.from_correlation(0.3, 0.6).pattern_probabilities(4)
    for row in rescored:
        words = TEXTS[int(row["message_id"])].split()[:8]
        table = BertScoreTable(JaccardScorer(), Message(0, " ".join(words), tuple(words)))
        partition = tuple(map(tuple, json.loads(row["partition"])))
        expected = float(table.lookup(union_masks(partition)) @ Q)
        assert float(row["ats"]) == pytest.approx(expected)

    assert (tmp_path / "results" / "bs" / "summary_bertscore.csv").exists()


def test_table_cache_resumes(tmp_path):
    words = TEXTS[1].split()[:8]
    message = Message(1, " ".join(words), tuple(words))
    cache = tmp_path / "c" / "1.json"

    first = JaccardScorer()
    table = BertScoreTable(first, message, cache_path=cache)
    a = table.lookup([0b101, 0b111])
    table.save()

    second = JaccardScorer()
    resumed = BertScoreTable(second, message, cache_path=cache)
    np.testing.assert_allclose(resumed.lookup([0b101, 0b111]), a)
    assert second.pairs == 0
