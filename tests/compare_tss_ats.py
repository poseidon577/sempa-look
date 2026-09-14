from sempalab.algorithms.greedy_tss import greedy_tss
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


message = Message(
    message_id=9,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

p = 0.3

# --------------------------------------------------
# Step 1: Construct partition using Greedy-TSS
# --------------------------------------------------

tss_result = greedy_tss(
    message=message,
    encoder=encoder,
    M=2,
)

partition = tss_result.best_partition

print("Greedy-TSS partition:")
print(partition)

print("TSS selection score:")
print(tss_result.best_score)


# --------------------------------------------------
# Step 2: Evaluate the resulting partition using ATS
# --------------------------------------------------

ats = exact_ats(
    partition,
    message,
    encoder,
    p,
)

print("\nATS of Greedy-TSS partition:")
print(ats)


assert partition is not None
assert 0.0 <= ats <= 1.0

print("\n✅ Greedy-TSS ATS evaluation passed")