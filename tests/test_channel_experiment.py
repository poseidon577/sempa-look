import json

import numpy as np
import pytest
import yaml

from sempalab.channel.correlated import GilbertElliott
from sempalab.channel_aware import channel_ats
from sempalab.channel_experiment import (
    ALGORITHMS,
    channel_points,
    load_config,
    read_csv,
    rescore,
    run,
    summarize,
)
from sempalab.evaluation import SimilarityTable
from sempalab.partition import validate_partition
from sempalab.types import Message


TEXTS = [
    "a man rides a red bike near the sea",
    "two dogs play with a ball in green grass",
    "a plate of food with rice and fresh greens",
]


@pytest.fixture
def workspace(tmp_path):
    eval_dir = tmp_path / "eval_sets"
    eval_dir.mkdir()
    (eval_dir / "coco_val2017_K8_n3_seed0.txt").write_text("\n".join(" ".join(t.split()[:8]) for t in TEXTS))

    def make_config(**fields):
        raw = {
            "data": {
                "source": "coco",
                "path": str(tmp_path / "captions_val2017.json"),
                "num_messages": 3,
                "seed": 0,
                "eval_set_dir": str(eval_dir),
            },
            "results_dir": str(tmp_path / "results"),
            "encoder": "minilm",
            "K": 8,
            "rayleigh_samples": 20_000,
            "r2_samples": 50,
            **fields,
        }
        path = tmp_path / f"{fields.get('experiment', 'exp')}.yaml"
        path.write_text(yaml.safe_dump(raw))
        return load_config(path)

    return make_config, tmp_path


CHANNELS = [{"name": "iid"}, {"name": "gilbert", "parameter": 0.8}, {"name": "rayleigh", "parameter": 0.9}]


def test_config_validation(workspace):
    make_config, _ = workspace

    with pytest.raises(ValueError):
        make_config(bogus=1)
    with pytest.raises(ValueError):
        make_config(M=[3])
    with pytest.raises(ValueError):
        make_config(algorithms=["nope"])
    with pytest.raises(ValueError):
        make_config(channels=[{"name": "awgn"}])
    with pytest.raises(ValueError):
        make_config(p=[0.0])


def test_channel_points(workspace):
    make_config, _ = workspace
    config = make_config(channels=CHANNELS, p=[0.1, 0.5])
    points = channel_points(config)
    assert len(points) == 6
    assert {"channel": "gilbert", "parameter": 0.8, "p": 0.5} in points


def test_run_end_to_end(workspace, fake_encoder):
    make_config, tmp_path = workspace
    config = make_config(experiment="e2e", M=[2, 4], p=[0.2, 0.5], channels=CHANNELS, stride=[1, 3], P=4, k=1)

    run(config, encoder=fake_encoder())

    rows = read_csv(tmp_path / "results" / "e2e" / "results.csv.gz")

    # per message and M: iid (1 stride) + 2 bursty channels x 2 strides, x 2 p values
    settings = (1 + 2 * 2) * 2
    assert len(rows) == 3 * 2 * settings * len(ALGORITHMS)

    by_key = {}
    for row in rows:
        partition = json.loads(row["partition"])
        assert validate_partition(tuple(map(tuple, partition)), 8, int(row["M"]))
        by_key[(row["message_id"], row["M"], row["channel"], row["stride"], row["p"], row["algorithm"])] = float(row["ats"])

    for key, value in by_key.items():
        message_id, M, channel, stride, p, name = key
        oracle = by_key[(message_id, M, channel, stride, p, "full_search")]
        assert value <= oracle + 1e-9, key

        # Reordering slots never hurts; under i.i.d. it changes nothing.
        if name in {"random_pa", "interleave", "sempa_look", "swap_ls"}:
            ordered = by_key[(message_id, M, channel, stride, p, f"{name}+order")]
            assert ordered >= value - 1e-9
            if channel == "iid":
                assert ordered == pytest.approx(value)

        if channel == "iid":
            assert by_key[(message_id, M, channel, stride, p, "full_search_iid")] == pytest.approx(oracle)

    summary = read_csv(tmp_path / "results" / "e2e" / "summary.csv")
    casa_rows = [r for r in summary if r["algorithm"] == "casa"]
    assert casa_rows and all(r["mean_surrogate_r2"] for r in casa_rows)
    assert all("gain_vs_sempa_look" in r for r in casa_rows)


def test_rows_score_exact_channel_ats(workspace, fake_encoder):
    make_config, tmp_path = workspace
    config = make_config(experiment="exact", M=[2], p=[0.4], channels=[{"name": "gilbert", "parameter": 0.7}],
                         algorithms=["random_pa", "casa", "full_search"])
    encoder = fake_encoder()
    run(config, encoder=encoder)

    rows = read_csv(tmp_path / "results" / "exact" / "results.csv.gz")
    Q = GilbertElliott.from_correlation(0.4, 0.7).pattern_probabilities(4)

    for row in rows:
        words = TEXTS[int(row["message_id"])].split()[:8]
        table = SimilarityTable(fake_encoder(), Message(int(row["message_id"]), " ".join(words), tuple(words)))
        partition = tuple(map(tuple, json.loads(row["partition"])))
        assert float(row["ats"]) == pytest.approx(channel_ats(table, partition, Q))


def test_run_is_reproducible(workspace, fake_encoder):
    make_config, tmp_path = workspace
    common = dict(M=[2], p=[0.3], channels=CHANNELS[:2], algorithms=["random_pa", "sempa_look", "swap_ls", "casa"], P=4, k=1)

    run(make_config(experiment="a", **common), encoder=fake_encoder())
    run(make_config(experiment="b", **common), encoder=fake_encoder())

    a = read_csv(tmp_path / "results" / "a" / "results.csv.gz")
    b = read_csv(tmp_path / "results" / "b" / "results.csv.gz")
    assert [r["partition"] for r in a] == [r["partition"] for r in b]


def test_rescore_keeps_partitions(workspace, fake_encoder):
    make_config, tmp_path = workspace
    config = make_config(experiment="rs", M=[4], p=[0.3], channels=CHANNELS[:2], algorithms=["random_pa", "casa"])
    run(config, encoder=fake_encoder())

    class OtherEncoder(fake_encoder):
        def _embed(self, tokens):
            return super()._embed(tokens[::-1] + tokens[:1])

    rescore(config, "bge-small", encoder=OtherEncoder())

    original = read_csv(tmp_path / "results" / "rs" / "results.csv.gz")
    rescored = read_csv(tmp_path / "results" / "rs" / "rescored_bge-small.csv.gz")

    assert [r["partition"] for r in original] == [r["partition"] for r in rescored]
    assert all(r["encoder"] == "bge-small" for r in rescored)
    assert any(abs(float(a["ats"]) - float(b["ats"])) > 1e-6 for a, b in zip(original, rescored))
    assert (tmp_path / "results" / "rs" / "summary_bge-small.csv").exists()


def test_summarize_paired_gain():
    rows = []
    for message_id, (a, b) in enumerate([(0.5, 0.6), (0.7, 0.75), (0.9, 0.95)]):
        for name, ats in (("sempa_look", a), ("casa", b)):
            rows.append({"K": 8, "M": 2, "channel": "iid", "parameter": 0.0, "stride": 1, "p": 0.3,
                         "algorithm": name, "message_id": message_id, "ats": ats, "logical_calls": 1, "encoder": "x"})

    casa_row = next(r for r in summarize(rows) if r["algorithm"] == "casa")
    assert casa_row["gain_vs_sempa_look"] == pytest.approx(np.mean([0.1, 0.05, 0.05]))
    assert casa_row["gain_vs_sempa_look_ci_low"] <= casa_row["gain_vs_sempa_look"] <= casa_row["gain_vs_sempa_look_ci_high"]


def test_shared_pool_keeps_calls_and_validity(workspace, fake_encoder):
    make_config, tmp_path = workspace
    config = make_config(experiment="pool", M=[2], p=[0.2, 0.5], channels=CHANNELS[:2],
                         algorithms=["casa", "full_search"], shared_pool=True)
    run(config, encoder=fake_encoder())
    rows = read_csv(tmp_path / "results" / "pool" / "results.csv.gz")
    for row in rows:
        assert validate_partition(tuple(map(tuple, json.loads(row["partition"]))), 8, 2)
    casa_calls = {int(r["logical_calls"]) for r in rows if r["algorithm"] == "casa"}
    assert min(casa_calls) > 0
