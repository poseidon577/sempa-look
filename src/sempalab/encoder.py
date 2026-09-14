from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class InstrumentedEncoder:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        batch_size=64,
        cache_size=200_000,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.device = device

        print(f"Loading encoder: {model_name}")
        print(f"Device: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        # Cache for arbitrary token-subset embeddings
        self.cache = OrderedDict()

        # Dedicated cache for the complete original message
        self.original_cache = {}

        # Instrumentation
        self.logical_calls = 0
        self.miss_calls = 0
        self.cache_hits = 0
        self.encoder_batches = 0
        self.wall_seconds = 0.0

    # ---------------------------------------------------------
    # Generic embedding cache
    # ---------------------------------------------------------

    def _cache_get(self, key):
        if key not in self.cache:
            return None

        value = self.cache.pop(key)
        self.cache[key] = value

        return value

    def _cache_put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)

        self.cache[key] = value

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    # ---------------------------------------------------------
    # Original-message embedding
    # ---------------------------------------------------------

    def embed_original(
        self,
        message_id,
        tokens,
    ):
        """
        Return the embedding of the complete original message.

        This is cached separately because every semantic-similarity
        evaluation compares a reconstruction against the same
        original message.
        """

        key = message_id

        if key in self.original_cache:
            self.cache_hits += 1
            return self.original_cache[key]

        text = " ".join(tokens)

        start = time.perf_counter()

        self.encoder_batches += 1

        embedding = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        self.wall_seconds += (
            time.perf_counter() - start
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        self.original_cache[key] = embedding

        self.miss_calls += 1

        return embedding

    # ---------------------------------------------------------
    # Arbitrary subset embeddings
    # ---------------------------------------------------------

    def embed_sets(self, index_sets, message_id, tokens):
        if not index_sets:
            return np.empty((0, 0), dtype=np.float32)

        start = time.perf_counter()

        # --------------------------------------------------------
        # 1. Canonicalize all requested index sets
        # --------------------------------------------------------

        canonical_sets = [
            tuple(sorted(indices))
            for indices in index_sets
        ]

        # --------------------------------------------------------
        # 2. Look up everything already in the cache
        # --------------------------------------------------------

        results = [None] * len(canonical_sets)

        missing_keys = []
        missing_texts = []
        missing_key_to_positions = {}

        for position, indices in enumerate(canonical_sets):

            key = (message_id, indices)

            cached = self._cache_get(key)

            if cached is not None:
                results[position] = cached
                self.cache_hits += 1
                continue

            # ----------------------------------------------------
            # This subset is not cached.
            #
            # But it may already have appeared earlier in THIS
            # batch. Avoid encoding it twice.
            # ----------------------------------------------------

            if key not in missing_key_to_positions:
                missing_key_to_positions[key] = []
                missing_keys.append(key)

                text = " ".join(
                    tokens[i]
                    for i in indices
                )

                missing_texts.append(text)

            missing_key_to_positions[key].append(position)

        # --------------------------------------------------------
        # 3. Encode each UNIQUE missing subset exactly once
        # --------------------------------------------------------

        if missing_texts:

            self.encoder_batches += int(
                np.ceil(len(missing_texts) / self.batch_size)
            )

            embeddings = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # miss_calls = number of UNIQUE embeddings computed
            self.miss_calls += len(missing_texts)

            # ----------------------------------------------------
            # 4. Put embeddings into cache and restore duplicates
            # ----------------------------------------------------

            for key, embedding in zip(
                missing_keys,
                embeddings,
            ):

                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

                self._cache_put(
                    key,
                    embedding,
                )

                # Same subset may have appeared multiple times.
                for position in missing_key_to_positions[key]:
                    results[position] = embedding

        self.wall_seconds += (
            time.perf_counter() - start
        )

        return np.stack(results)