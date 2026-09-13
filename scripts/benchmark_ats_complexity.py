from math import log2


def exact_ats_evaluations(N):
    return 2 ** N


print("Exact ATS complexity")
print("=" * 40)

for N in [2, 3, 5, 10, 15, 20]:
    evaluations = exact_ats_evaluations(N)

    print(
        f"N={N:2d} packets | "
        f"evaluations={evaluations:,}"
    )

print("\nGrowth:")
for N in [10, 15, 20]:
    evaluations = exact_ats_evaluations(N)

    print(
        f"N={N:2d}: "
        f"2^{N} = {evaluations:,}"
    )