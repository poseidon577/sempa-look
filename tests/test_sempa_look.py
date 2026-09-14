from sempalab.algorithms.sempa_look import sempa_look
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


message = Message(
    message_id=11,
    text="a small motor bike",
    tokens=("a", "small", "motor", "bike"),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

M = 2
P = 10
k = 1
p = 0.3
seed = 42


result = sempa_look(
    message=message,
    encoder=encoder,
    M=M,
    P=P,
    k=k,
    seed=seed,
)

partition = result.best_partition


print("SemPA-Look partition:")
print(partition)

print("\nSemPA-Look selection score:")
print(result.best_score)


# ---------------------------------------------------------
# Structural checks
# ---------------------------------------------------------

assert len(partition) == len(message.tokens) // M

assert all(
    len(group) == M
    for group in partition
)

all_indices = [
    index
    for group in partition
    for index in group
]

assert sorted(all_indices) == list(
    range(len(message.tokens))
)

assert len(set(all_indices)) == len(
    message.tokens
)


# ---------------------------------------------------------
# Complexity check
#
# K=4, M=2 -> N=2
#
# SemPA-Look performs N-1 = 1 decision depth.
#
# At that depth:
#
# P(k+1) = 10 * 2 = 20
#
# logical semantic evaluations.
# ---------------------------------------------------------

expected_logical_calls = (len(message.tokens) // M - 1) * P * (k + 1)

assert result.logical_calls == expected_logical_calls, (
    f"Expected {expected_logical_calls} logical calls, "
    f"got {result.logical_calls}"
)


# ---------------------------------------------------------
# Common ATS evaluation
# ---------------------------------------------------------

ats = exact_ats(
    partition=partition,
    message=message,
    encoder=encoder,
    p=p,
)

print("\nATS of SemPA-Look partition:")
print(ats)


assert 0.0 <= ats <= 1.0


print("\nInstrumentation:")
print(f"Logical calls: {result.logical_calls}")
print(f"Miss calls: {result.miss_calls}")
print(f"Cache hits: {result.cache_hits}")
print(f"Runtime: {result.runtime_seconds}")


print("\n✅ SemPA-Look test passed")