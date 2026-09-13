from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import semantic_similarity
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


encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


# --------------------------------------------------
# First evaluation
# --------------------------------------------------

score_1 = semantic_similarity(
    encoder,
    message,
    surviving_indices=(1, 3),
)

print("First score:", score_1)
print("Logical calls:", encoder.logical_calls)
print("Miss calls:", encoder.miss_calls)
print("Cache hits:", encoder.cache_hits)


# --------------------------------------------------
# Same evaluation again
# --------------------------------------------------

score_2 = semantic_similarity(
    encoder,
    message,
    surviving_indices=(1, 3),
)

print("\nSecond score:", score_2)
print("Logical calls:", encoder.logical_calls)
print("Miss calls:", encoder.miss_calls)
print("Cache hits:", encoder.cache_hits)


# --------------------------------------------------
# Assertions
# --------------------------------------------------

assert 0 < score_1 < 1
assert abs(score_1 - score_2) < 1e-6

# Second evaluation should hit the cache.
assert encoder.cache_hits >= 2

# We requested 4 embeddings logically:
#
# first call:
#   original + subset = 2
#
# second call:
#   original + subset = 2
#
assert encoder.logical_calls == 4

# But only the first two should require
# actual encoder computations.
assert encoder.miss_calls == 2


print("\n✅ Semantic metric + cache test passed")