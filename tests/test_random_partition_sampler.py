from collections import Counter

import numpy as np
from scipy.stats import chisquare

from sempalab.partition import (
    generate_partitions,
    sample_random_partition,
    validate_partition,
)


def test_samples_are_valid_and_canonical():
    rng = np.random.default_rng(0)

    canonical = set(generate_partitions(12, 4))

    for _ in range(200):
        partition = sample_random_partition(12, 4, rng)

        assert validate_partition(partition, 12, 4)
        assert partition in canonical


def test_sampler_is_uniform_over_the_35_partitions():
    rng = np.random.default_rng(1)

    n = 35_000
    counts = Counter(sample_random_partition(8, 4, rng) for _ in range(n))

    assert len(counts) == 35

    _, p_value = chisquare(list(counts.values()))

    assert p_value > 1e-3


def test_large_K_does_not_enumerate():
    # 24!/((4!)^6 6!) ~ 4.5e12 partitions: must not enumerate.
    partition = sample_random_partition(24, 4, np.random.default_rng(2))

    assert validate_partition(partition, 24, 4)
