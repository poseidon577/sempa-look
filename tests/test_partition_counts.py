import math


def partition_count(K, M):
    N = K // M

    assert K % M == 0

    return math.factorial(K) // (
        math.factorial(M) ** N
        * math.factorial(N)
    )


expected = {
    (8, 4): 35,
    (9, 3): 280,
    (12, 4): 5775,
    (12, 3): 15400,
    (12, 2): 10395,
    (16, 4): 2627625,
}


for (K, M), expected_count in expected.items():

    actual = partition_count(K, M)

    print(
        f"K={K}, M={M}, "
        f"N={K // M} | "
        f"actual={actual:,} | "
        f"expected={expected_count:,}"
    )

    assert actual == expected_count


print("\n✅ Partition counting test passed")