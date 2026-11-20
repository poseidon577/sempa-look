import json

import numpy as np
import pytest
import yaml

from sempalab.channel.correlated import GilbertElliott, GaussMarkovRayleigh, iid_pattern_probabilities
from sempalab.channel_experiment import load_config as load_main_config
from sempalab.channel_experiment import read_csv
from sempalab.channel_experiment import run as run_main
from sempalab.channel_mismatch import (
    design_label,
    empirical_law,
    estimate_gilbert,
    load_config,
    run,
)
from sempalab.partition import validate_partition


TEXTS = [
    "a man rides a red bike near the sea",
    "two dogs play with a ball in green grass",
    "a plate of food with rice and fresh greens",
]


# ---------------------------------------------------------------- estimators


def test_estimate_gilbert_recovers_parameters():
    truth = GilbertElliott.from_correlation(0.3, 0.6)
    trace = truth.sample_erasures(200_000, np.random.default_rng(0))
    fitted = estimate_gilbert(trace)
    assert fitted.p == pytest.approx(0.3, abs=0.01)
    assert fitted.parameter == pytest.approx(0.6, abs=0.02)
    np.testing.assert_allclose(fitted.pattern_probabilities(4), truth.pattern_probabilities(4), atol=0.01)


def test_estimate_gilbert_hand_counts():
    # transitions: G->B once, B->B once, B->G once, G->G twice (+0.5 each)
    trace = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    fitted = estimate_gilbert(trace, smoothing=0.5)
    assert fitted.p_gb == pytest.approx(1.5 / 4.0)
    assert fitted.p_bg == pytest.approx(1.5 / 3.0)


def test_estimate_gilbert_needs_two_packets():
    with pytest.raises(ValueError):
        estimate_gilbert(np.array([True]))


def test_empirical_law_hand_counts():
    # windows of N=2, stride 1: (s,s) (s,e) (e,s) -> survive bits: slot0 bit0, slot1 bit1
    trace = np.array([0, 0, 1, 0], dtype=bool)
    law = empirical_law(trace, N=2, stride=1, smoothing=0.0)
    # (0,0)->both survive->3, (0,1)->slot0 only->1, (1,0)->slot1 only->2
    np.testing.assert_allclose(law, [0, 1 / 3, 1 / 3, 1 / 3])


def test_empirical_law_converges_with_stride():
    truth = GilbertElliott.from_correlation(0.4, 0.8)
    trace = truth.sample_erasures(300_000, np.random.default_rng(1))
    np.testing.assert_allclose(empirical_law(trace, 3, stride=2), truth.pattern_probabilities(3, stride=2), atol=0.01)


def test_empirical_law_iid():
    rng = np.random.default_rng(2)
    trace = rng.random(400_000) < 0.2
    np.testing.assert_allclose(empirical_law(trace, 3), iid_pattern_probabilities(3, 0.2), atol=0.005)


def test_empirical_law_rejects_short_trace():
    with pytest.raises(ValueError):
        empirical_law(np.zeros(3, dtype=bool), N=3, stride=2)


def test_fitted_gilbert_on_fading_is_a_valid_law():
    truth = GaussMarkovRayleigh(0.3, 0.9, samples=20_000)
    fitted = estimate_gilbert(truth.sample_erasures(50_000, np.random.default_rng(3)))
    Q = fitted.pattern_probabilities(4)
    assert Q.sum() == pytest.approx(1.0)
    assert fitted.p == pytest.approx(0.3, abs=0.03)


def test_design_labels_unique():
    labels = [design_label(d) for d in (
        {"kind": "matched"}, {"kind": "iid"}, {"kind": "gilbert", "parameter": 0.9},
        {"kind": "rayleigh", "parameter": 0.99}, {"kind": "p_scale", "factor": 0.5},
        {"kind": "estimate_gilbert", "packets": 100}, {"kind": "estimate_empirical", "packets": 100})]
    assert len(set(labels)) == len(labels)


# ---------------------------------------------------------------- runner


@pytest.fixture
def workspace(tmp_path):
    eval_dir = tmp_path / "eval_sets"
    eval_dir.mkdir()
    (eval_dir / "coco_val2017_K8_n3_seed0.txt").write_text("\n".join(" ".join(t.split()[:8]) for t in TEXTS))

    def config_dict(**fields):
        return {
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
            "P": 4,
            "k": 1,
            **fields,
        }

    def make(loader, name, **fields):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config_dict(experiment=name, **fields)))
        return loader(path)

    return make, tmp_path


DESIGNS = [
    {"kind": "matched"},
    {"kind": "iid"},
    {"kind": "gilbert", "parameter": 0.9},
    {"kind": "rayleigh", "parameter": 0.9},
    {"kind": "p_scale", "factor": 1.5},
    {"kind": "estimate_gilbert", "packets": 200},
    {"kind": "estimate_empirical", "packets": 200},
]


def test_config_validation(workspace):
    make, _ = workspace
    with pytest.raises(ValueError):
        make(load_config, "bad1", designs=[{"kind": "iid"}])  # no matched
    with pytest.raises(ValueError):
        make(load_config, "bad2", designs=[{"kind": "matched"}, {"kind": "nope"}])
    with pytest.raises(ValueError):
        make(load_config, "bad3", designs=[{"kind": "matched"}, {"kind": "matched"}])


def test_run_end_to_end(workspace, fake_encoder):
    make, tmp_path = workspace
    channels = [{"name": "gilbert", "parameter": 0.6}, {"name": "rayleigh", "parameter": 0.9}]
    config = make(load_config, "mm", M=[2, 4], p=[0.3], channels=channels, designs=DESIGNS)
    run(config, encoder=fake_encoder())

    rows = read_csv(tmp_path / "results" / "mm" / "results.csv.gz")
    per_setting = 4 + 2 * len(DESIGNS)  # random_pa, sempa_look, swap_ls, full_search + 2 per design
    assert len(rows) == 3 * 2 * len(channels) * per_setting

    by_key = {}
    for row in rows:
        assert validate_partition(tuple(map(tuple, json.loads(row["partition"]))), 8, int(row["M"]))
        by_key[(row["message_id"], row["M"], row["channel"], row["algorithm"])] = row

    for (message_id, M, channel, name), row in by_key.items():
        oracle = float(by_key[(message_id, M, channel, "full_search")]["ats"])
        assert float(row["ats"]) <= oracle + 1e-9
        if name.endswith("@matched"):
            assert float(row["tv_distance"]) == pytest.approx(0.0)
        # A design never beats ordering the same partition for the true law.
        if name.startswith("swap_ls+order@"):
            assert float(row["ats"]) <= float(by_key[(message_id, M, channel, "swap_ls+order@matched")]["ats"]) + 1e-9

    summary = read_csv(tmp_path / "results" / "mm" / "summary.csv")
    casa_iid = [r for r in summary if r["algorithm"] == "casa@iid"]
    assert casa_iid and all("gain_vs_casa@matched" in r and r["share_of_oracle_gain"] for r in casa_iid)


def test_matched_design_reproduces_main_casa(workspace, fake_encoder):
    make, tmp_path = workspace
    common = dict(M=[2], p=[0.4], channels=[{"name": "gilbert", "parameter": 0.7}])
    run(make(load_config, "mm2", designs=[{"kind": "matched"}], **common), encoder=fake_encoder())
    run_main(make(load_main_config, "main2", algorithms=["casa"], **common), encoder=fake_encoder())

    mismatch = {r["message_id"]: r["partition"] for r in read_csv(tmp_path / "results" / "mm2" / "results.csv.gz") if r["algorithm"] == "casa@matched"}
    main = {r["message_id"]: r["partition"] for r in read_csv(tmp_path / "results" / "main2" / "results.csv.gz") if r["algorithm"] == "casa"}
    assert mismatch == main


def test_estimated_designs_are_reproducible(workspace, fake_encoder):
    make, tmp_path = workspace
    common = dict(M=[2], p=[0.3], channels=[{"name": "gilbert", "parameter": 0.6}], designs=DESIGNS[:1] + DESIGNS[-2:])
    run(make(load_config, "r1", **common), encoder=fake_encoder())
    run(make(load_config, "r2", **common), encoder=fake_encoder())
    a = read_csv(tmp_path / "results" / "r1" / "results.csv.gz")
    b = read_csv(tmp_path / "results" / "r2" / "results.csv.gz")
    assert [(r["partition"], r["tv_distance"]) for r in a] == [(r["partition"], r["tv_distance"]) for r in b]
