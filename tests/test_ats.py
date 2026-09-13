from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


message = Message(
    message_id=0,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)


partition = (
    (0, 2),
    (1, 3),
)


encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


# --------------------------------------------------
# p = 0
# --------------------------------------------------
# No packet should be lost.
# Therefore ATS should equal 1.

ats_no_loss = exact_ats(
    partition,
    message,
    encoder,
    p=0.0,
)

print("ATS at p=0:", ats_no_loss)

assert abs(ats_no_loss - 1.0) < 1e-6


# --------------------------------------------------
# p = 1
# --------------------------------------------------
# Every packet is lost.
# Reconstruction is empty.
#
# ATS should equal phi(empty, W).

ats_total_loss = exact_ats(
    partition,
    message,
    encoder,
    p=1.0,
)

print("ATS at p=1:", ats_total_loss)

assert 0.0 <= ats_total_loss <= 1.0


print("\nLogical calls:", encoder.logical_calls)
print("Miss calls:", encoder.miss_calls)
print("Cache hits:", encoder.cache_hits)

print("\n✅ Exact ATS test passed")