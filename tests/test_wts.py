from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import wts
from sempalab.types import Message


message = Message(
    message_id=2,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)

partition = (
    (0,),
    (1,),
    (2,),
    (3,),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


# --------------------------------------------------
# p = 0
# --------------------------------------------------
# No packet loss is possible.
# Therefore WTS must equal similarity of the
# complete message with itself.

wts_no_loss = wts(
    partition,
    message,
    encoder,
    p=0.0,
)

print("WTS at p=0:", wts_no_loss)

assert abs(wts_no_loss - 1.0) < 1e-6


# --------------------------------------------------
# p = 1
# --------------------------------------------------
# Every packet is lost.
# Therefore WTS must equal similarity of the
# empty reconstruction.

wts_total_loss = wts(
    partition,
    message,
    encoder,
    p=1.0,
)

print("WTS at p=1:", wts_total_loss)

assert 0.0 <= wts_total_loss <= 1.0


print("\nLogical calls:", encoder.logical_calls)
print("Miss calls:", encoder.miss_calls)
print("Cache hits:", encoder.cache_hits)

print("\n✅ WTS edge-case test passed")