import numpy as np

from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats, monte_carlo_ats
from sempalab.types import Message


message = Message(
    message_id=1,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)


# Four packets means 2^4 = 16 possible
# survival states, which is small enough
# for exact ATS.
partition = (
    (0,),
    (1,),
    (2,),
    (3,),
)


encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


p = 0.3


exact = exact_ats(
    partition,
    message,
    encoder,
    p,
)


monte_carlo = monte_carlo_ats(
    partition,
    message,
    encoder,
    p,
    samples=10_000,
    rng=np.random.default_rng(42),
)


print("Exact ATS:", exact)
print("Monte Carlo ATS:", monte_carlo)
print("Absolute difference:", abs(exact - monte_carlo))


# Monte Carlo should be close, but not identical.
assert abs(exact - monte_carlo) < 0.02


print("\n✅ Monte Carlo ATS test passed")