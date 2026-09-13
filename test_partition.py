from partition import generate_partitions, validate_partition


def test_k4_m2():
    partitions = list(generate_partitions(4, 2))

    expected = {
        (
            (0, 1),
            (2, 3),
        ),
        (
            (0, 2),
            (1, 3),
        ),
        (
            (0, 3),
            (1, 2),
        ),
    }

    actual = set(partitions)

    assert actual == expected

    for partition in partitions:
        assert validate_partition(
            partition,
            K=4,
            M=2,
        )


def test_counts():
    expected_counts = {
        (8, 4): 35,
        (9, 3): 280,
        (12, 4): 5775,
        (12, 3): 15400,
        (12, 2): 10395,
    }

    for (K, M), expected in expected_counts.items():

        partitions = list(
            generate_partitions(K, M)
        )

        assert len(partitions) == expected

        for partition in partitions:
            assert validate_partition(
                partition,
                K,
                M,
            )


if __name__ == "__main__":
    test_k4_m2()
    test_counts()

    print("✅ Partition generator tests passed")
    