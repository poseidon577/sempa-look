import zlib

import numpy as np
import pytest


class CachingFakeEncoder:
    """
    Torch-free stand-in with the InstrumentedEncoder interface,
    including a (message_id, indices) cache.
    """

    def __init__(self, dim=32):
        self.dim = dim
        self.logical_calls = 0
        self.miss_calls = 0
        self.cache_hits = 0
        self.encoder_batches = 0
        self.cache = {}

    def clear_cache(self):
        self.cache.clear()

    def _embed(self, tokens):
        vector = np.zeros(self.dim)
        for token in tokens:
            vector += np.random.default_rng(zlib.crc32(token.encode())).standard_normal(self.dim)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def embed_original(self, message_id, tokens):
        return self._embed(tokens)

    def embed_sets(self, index_sets, message_id, tokens):
        out = []
        for indices in index_sets:
            key = (message_id, tuple(sorted(indices)))
            if key in self.cache:
                self.cache_hits += 1
            else:
                self.miss_calls += 1
                self.cache[key] = self._embed([tokens[i] for i in key[1]])
            out.append(self.cache[key])
        self.encoder_batches += 1
        return np.stack(out)


@pytest.fixture
def fake_encoder():
    """
    Factory for a deterministic, torch-free encoder with a
    (message_id, indices) cache.
    """

    return CachingFakeEncoder
