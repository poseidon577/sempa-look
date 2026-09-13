from itertools import combinations


def validate_partition(
    partition: tuple[tuple[int, ...], ...],
    K: int,
    M: int,
) -> bool:
    """
    Validate that `partition` divides token indices 0..K-1
    into N non-overlapping groups, each containing M indices.
    """

    # K must be divisible by M.
    assert K % M == 0

    N = K // M

    # Correct number of groups.
    if len(partition) != N:
        return False

    # Every group must have exactly M elements.
    if any(len(group) != M for group in partition):
        return False

    # Flatten all indices.
    indices = [i for group in partition for i in group]

    # No duplicates and complete coverage of 0..K-1.
    if sorted(indices) != list(range(K)):
        return False

    return True


def generate_partitions(
    K: int,
    M: int,
):
    """
    Generate all unlabeled partitions of {0,...,K-1}
    into groups of size M.

    Groups are represented canonically: the first group
    always contains the smallest currently-unassigned index.
    """

    assert K % M == 0

    remaining = tuple(range(K))

    def recurse(
        remaining: tuple[int, ...],
        groups: tuple[tuple[int, ...], ...],
    ):
        # Nothing left to assign.
        if not remaining:
            yield groups
            return

        # Canonical choice:
        # the smallest remaining index MUST belong
        # to the next group.
        first = remaining[0]

        # Choose the other M-1 members of this group.
        for others in combinations(remaining[1:], M - 1):

            group = tuple(sorted((first, *others)))

            # Remove this group's indices.
            group_set = set(group)

            new_remaining = tuple(
                i for i in remaining
                if i not in group_set
            )

            yield from recurse(
                new_remaining,
                groups + (group,),
            )

    yield from recurse(remaining, ())