import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Original message
# --------------------------------------------------

tokens = ["a", "small", "motor", "bike"]

print("Original tokens:", tokens)


# --------------------------------------------------
# 2. Create two arbitrary subpacket groups
# --------------------------------------------------

group_1 = (0, 2)  # a, motor
group_2 = (1, 3)  # small, bike

print("Group 1:", group_1)
print("Group 2:", group_2)


# --------------------------------------------------
# 3. Reconstruct after losing group 1
# --------------------------------------------------

reconstructed_tokens = [
    tokens[i]
    for i in sorted(group_2)
]

reconstructed = " ".join(reconstructed_tokens)

original = " ".join(tokens)

print("Original:", original)
print("Reconstructed:", reconstructed)


# --------------------------------------------------
# 4. Choose device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# --------------------------------------------------
# 5. Load MiniLM
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)


# --------------------------------------------------
# 6. Encode original and reconstructed messages
# --------------------------------------------------

embeddings = model.encode(
    [original, reconstructed],
    convert_to_numpy=True,
    normalize_embeddings=True
)

original_embedding = embeddings[0]
reconstructed_embedding = embeddings[1]


# --------------------------------------------------
# 7. Cosine similarity
# --------------------------------------------------

similarity = cosine_similarity(
    [original_embedding],
    [reconstructed_embedding]
)[0][0]

print("Semantic similarity:", similarity)


# --------------------------------------------------
# 8. Assertions
# --------------------------------------------------

assert 0 < similarity < 1

print("✅ Vertical slice passed")