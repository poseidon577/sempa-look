import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from sempalab.aggregation import aggregate_into_packets
from sempalab.config import load_config
from sempalab.run import run
from sempalab.system import simulate_batch_ats, transmit_batch
from sempalab.types import Message


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FakeEncoder:
    """
    Deterministic torch-free stand-in for InstrumentedEncoder:
    a message embeds as the normalized sum of hashed token vectors.
    """

    def __init__(self, dim=32):
        self.dim = dim
        self.logical_calls = 0
        self.miss_calls = 0
        self.cache_hits = 0
        self.encoder_batches = 0

    def _embed(self, tokens):
        vector = np.zeros(self.dim)

        for token in tokens:
            rng = np.random.default_rng(zlib.crc32(token.encode()))
            vector += rng.standard_normal(self.dim)

        norm = np.linalg.norm(vector)

        return vector / norm if norm > 0 else vector

    def embed_original(self, message_id, tokens):
        return self._embed(tokens)

    def embed_sets(self, index_sets, message_id, tokens):
        self.miss_calls += len(index_sets)
        self.encoder_batches += 1

        return [
            self._embed([tokens[i] for i in indices])
            for indices in index_sets
        ]


def make_messages(B):
    words = "a man riding a bicycle near the beach".split()

    return [
        Message(message_id=b, text=" ".join(words), tokens=tuple(words))
        for b in range(B)
    ]


def test_transmit_batch_high_snr_delivers_everything():
    messages = make_messages(3)
    partitions = [((0, 1, 2, 3), (4, 5, 6, 7))] * 3

    surviving, erased = transmit_batch(
        messages, partitions, snr_db=30, rng=np.random.default_rng(0)
    )

    assert not erased.any()
    assert surviving == {b: tuple(range(8)) for b in range(3)}


def test_simulated_ats_tracks_exact_ats():
    from sempalab.metrics import exact_ats

    encoder = FakeEncoder()
    messages = make_messages(4)
    partitions = [((0, 2, 4, 6), (1, 3, 5, 7))] * 4

    ats, per = simulate_batch_ats(
        messages,
        partitions,
        encoder,
        snr_db=6,
        trials=4000,
        rng=np.random.default_rng(7),
    )

    from sempalab.channel.awgn_bpsk import packet_error_probability

    p = packet_error_probability(6)
    expected = exact_ats(partitions[0], messages[0], encoder, p)

    assert per == pytest.approx(p, abs=0.02)

    for value in ats.values():
        assert value == pytest.approx(expected, abs=0.03)


def test_demo_config_loads_with_base():
    config = load_config(CONFIG_DIR / "stage3_demo.yaml")

    assert config["experiment"] == "stage3_demo"
    assert config["packet_bits"] == 300
    assert config["dictionary_size"] == 200_000
    assert config["K"] == 8 and config["M"] == 4


def write_config(tmp_path, **overrides):
    config = {
        "experiment": "unit",
        "dataset": "inline",
        "messages": [
            "a man riding a bicycle near the beach",
            "two dogs play with a ball in grass",
            "too short",
            "a red bus parked on a city street",
        ],
        "K": 8,
        "M": 4,
        "B": 2,
        "P": 4,
        "k": 1,
        "snr_db": [6, 8],
        "algorithms": ["full_search", "random_pa", "greedy_rss", "sempa_look"],
        "trials": 50,
        "results_dir": str(tmp_path / "results"),
    }
    config.update(overrides)

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))

    return path


def test_run_end_to_end(tmp_path):
    config = load_config(write_config(tmp_path))

    summary = run(config, encoder=FakeEncoder())

    out_dir = tmp_path / "results" / "unit"
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "config.yaml").exists()

    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved == summary

    # 4 algorithms x 2 SNRs, 3 messages with exactly K=8 tokens.
    assert len(summary["results"]) == 8
    assert {row["messages"] for row in summary["results"]} == {3}

    exact = {
        (row["algorithm"], row["snr_db"]): row["mean_exact_ats"]
        for row in summary["results"]
    }

    # Full search is optimal for exact ATS at every SNR.
    for (name, snr_db), value in exact.items():
        assert exact[("full_search", snr_db)] >= value - 1e-9

    # Higher SNR -> fewer erasures -> higher ATS.
    assert exact[("full_search", 8)] > exact[("full_search", 6)]


def test_run_is_reproducible(tmp_path):
    config = load_config(write_config(tmp_path))

    first = run(config, encoder=FakeEncoder())
    second = run(config, encoder=FakeEncoder())

    # Wall-clock runtime is the only field allowed to differ.
    for summary in (first, second):
        for row in summary["results"]:
            row.pop("mean_runtime_seconds")

    assert first == second


@pytest.mark.parametrize(
    "overrides",
    [
        {"M": 3},
        {"k": 2},
        {"algorithms": ["nope"]},
        {"dataset": "ms_coco"},
        {"mode": "fast"},
        {"typo_key": 1},
    ],
)
def test_invalid_configs_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, **overrides))
