import csv
import json
from math import comb

import numpy as np
import pytest
import yaml

from sempalab.channel.awgn_bpsk import packet_error_probability
from sempalab.repro import (
    block_unions,
    count_block_unions,
    load_repro_config,
    run,
    sweep_points,
    union_states,
)


CAPTIONS = [
    "A man riding a bicycle near the beach.",
    "Two dogs play with a ball in grass.",
    "A plate of food with rice and vegetables.",
    "A red bus parked on a city street.",
    "A woman holding an umbrella in the rain.",
    "A cat sleeping on top of a laptop.",
    "A large clock tower stands in the city center today.",
]


@pytest.fixture
def workspace(tmp_path):
    annotations = tmp_path / "captions_val2017.json"
    annotations.write_text(
        json.dumps({"annotations": [{"id": i, "image_id": i, "caption": c} for i, c in enumerate(CAPTIONS)]})
    )

    def make_config(**fields):
        coco_override = fields.pop("coco", {})
        raw = {
            "coco": {
                "annotations": str(annotations),
                "num_messages": 4,
                "seed": 0,
                "eval_set_dir": str(tmp_path / "eval_sets"),
                **coco_override,
            },
            "results_dir": str(tmp_path / "results"),
            **fields,
        }
        path = tmp_path / f"{fields.get('experiment', 'exp')}.yaml"
        path.write_text(yaml.safe_dump(raw))
        return load_repro_config(path)

    return make_config, tmp_path


def test_union_states_cover_all_erasure_patterns():
    states = union_states(((0, 2), (1, 3)))

    assert sorted(states) == [(), (0, 1, 2, 3), (0, 2), (1, 3)]


@pytest.mark.parametrize("K, M", [(8, 4), (12, 4), (12, 3), (6, 2)])
def test_block_unions_count(K, M):
    unions = list(block_unions(K, M))

    assert len(unions) == count_block_unions(K, M) == sum(comb(K, s) for s in range(0, K + 1, M))
    assert all(len(u) % M == 0 for u in unions)


def test_config_rejects_unknown_keys(workspace):
    make_config, _ = workspace

    with pytest.raises(ValueError):
        make_config(bogus=1)


def test_config_rejects_bad_sweep(workspace):
    make_config, _ = workspace

    with pytest.raises(ValueError):
        make_config(K=8, sweep={"param": "M", "values": [3]})  # 3 does not divide 8


def test_sweep_points(workspace):
    make_config, _ = workspace

    config = make_config(K=8, M=4, sweep={"param": "M", "values": [1, 2, 4]})

    assert [point["M"] for point in sweep_points(config)] == [1, 2, 4]
    assert all(point["K"] == 8 for point in sweep_points(config))


def test_run_end_to_end(workspace, fake_encoder):
    make_config, tmp_path = workspace

    config = make_config(
        experiment="fig6_smoke",
        K=8,
        M=4,
        P=10,
        k=1,
        snr_db=[4, 8],
        algorithms=["full_search", "random_pa", "no_packetization", "greedy_rss", "sempa_look"],
    )

    summary = run(config, encoder=fake_encoder())

    out = tmp_path / "results" / "fig6_smoke"
    assert (out / "results.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "config.yaml").exists()

    rows = list(csv.DictReader((out / "results.csv").open()))
    assert len(rows) == 4 * 5 * 2  # messages x algorithms x SNRs

    by = {(r["algorithm"], r["snr_db"]): r for r in summary_rows(summary)}

    for snr in (4, 8):
        assert by[("full_search", snr)]["p"] == pytest.approx(packet_error_probability(snr))

        # Full search maximizes exact ATS, so it beats everything per message.
        best = {}
        for r in rows:
            if int(r["snr_db"]) == snr:
                best.setdefault(r["message_id"], {})[r["algorithm"]] = float(r["ats"])
        for scores in best.values():
            assert scores["full_search"] >= max(scores.values()) - 1e-9

    # SemPA-Look calls: (N-1) * P * (k+1) = 1 * 10 * 2
    assert by[("sempa_look", 4)]["mean_logical_calls"] == 20
    # Random and no-packetization cost no encoder calls.
    assert by[("random_pa", 4)]["mean_logical_calls"] == 0

    # p-independent algorithms keep the same partition across SNRs.
    for r4, r8 in zip(
        [r for r in rows if r["algorithm"] == "sempa_look" and r["snr_db"] == "4"],
        [r for r in rows if r["algorithm"] == "sempa_look" and r["snr_db"] == "8"],
    ):
        assert r4["partition"] == r8["partition"]

    # ATS and WTS rise with SNR, and WTS <= ATS.
    for name in ("random_pa", "sempa_look"):
        assert by[(name, 8)]["mean_ats"] > by[(name, 4)]["mean_ats"]
        assert by[(name, 8)]["mean_wts"] <= by[(name, 8)]["mean_ats"] + 1e-9


def test_run_is_reproducible(workspace, fake_encoder):
    make_config, _ = workspace

    config = make_config(K=8, M=4, k=1, snr_db=[6], algorithms=["random_pa", "sempa_look"])

    a = run(config, encoder=fake_encoder())
    b = run(config, encoder=fake_encoder())

    assert a == b


def test_sweep_over_K_loads_new_messages(workspace, fake_encoder):
    make_config, tmp_path = workspace

    # M=4 does not divide 9
    with pytest.raises(ValueError):
        make_config(
            experiment="k_sweep",
            K=8,
            M=4,
            sweep={"param": "K", "values": [8, 9]},
        )

    config = make_config(
        experiment="k_sweep",
        K=8,
        M=2,
        k=1,
        snr_db=[8],
        algorithms=["random_pa", "sempa_look"],
        sweep={"param": "K", "values": [8, 10]},
        coco={"num_messages": 1},
    )

    summary = run(config, encoder=fake_encoder())

    assert {row["K"] for row in summary} == {8, 10}
    assert (tmp_path / "eval_sets" / "coco_val2017_K10_n1_seed0.txt").exists()


def summary_rows(summary):
    return summary
