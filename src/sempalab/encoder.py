from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class InstrumentedEncoder:
    """
    Sentence-transformer wrapper with:

    - batched encoding
    - L2-normalized embeddings
    - embedding cache
    - logical-call accounting
    - actual model-forward accounting
    - cache-hit accounting
    - wall-clock accounting
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        cache_size: int = 200_000,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size

        # --------------------------------------------------
        # Choose device
        # --------------------------------------------------

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.device = device

        # --------------------------------------------------
        # Load encoder
        # --------------------------------------------------

        print(f"Loading encoder: {model_name}")
        print(f"Device: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        # --------------------------------------------------
        # Cache
        # --------------------------------------------------

        self.cache: OrderedDict[
            tuple[int, tuple[int, ...]],
            np.ndarray,
        ] = OrderedDict()

        # --------------------------------------------------
        # Counters
        # --------------------------------------------------

        self.logical_calls = 0
        self.miss_calls = 0
        self.cache_hits = 0
        self.wall_seconds = 0.0

    def _cache_get(
        self,
        key: tuple[int, tuple[int, ...]],
    ) -> np.ndarray | None:

        if key not in self.cache:
            return None

        # LRU behavior:
        # recently accessed item moves to the end.
        value = self.cache.pop(key)
        self.cache[key] = value

        return value

    def _cache_put(
        self,
        key: tuple[int, tuple[int, ...]],
        value: np.ndarray,
    ) -> None:

        if key in self.cache:
            self.cache.pop(key)

        self.cache[key] = value

        # Remove oldest entries if cache is full.
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def embed_sets(
        self,
        index_sets: list[tuple[int, ...]],
        message_id: int,
        tokens: tuple[str, ...],
    ) -> np.ndarray:
        """
        Encode many token subsets from one message.

        Parameters
        ----------
        index_sets:
            Token-index subsets to encode.

        message_id:
            Unique identifier of the message.

        tokens:
            Original token sequence.

        Returns
        -------
        np.ndarray
            Shape: (len(index_sets), embedding_dimension)
        """

        if not index_sets:
            return np.empty((0, 0), dtype=np.float32)

        # Every requested semantic evaluation counts
        # as one logical call.

        start = time.perf_counter()

        results: list[np.ndarray | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # --------------------------------------------------
        # Check cache
        # --------------------------------------------------

        for position, indices in enumerate(index_sets):

            canonical_indices = tuple(sorted(indices))

            key = (
                message_id,
                canonical_indices,
            )

            cached = self._cache_get(key)

            if cached is not None:
                results.append(cached)
                self.cache_hits += 1
            else:
                results.append(None)
                missing_indices.append(position)

                text = " ".join(
                    tokens[i]
                    for i in canonical_indices
                )

                missing_texts.append(text)

        # --------------------------------------------------
        # Encode cache misses in batches
        # --------------------------------------------------

        if missing_texts:

            embeddings = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.miss_calls += len(missing_texts)

            for position, embedding, text in zip(
                missing_indices,
                embeddings,
                missing_texts,
            ):
                indices = tuple(
                    sorted(index_sets[position])
                )

                key = (
                    message_id,
                    indices,
                )

                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

                self._cache_put(
                    key,
                    embedding,
                )

                results[position] = embedding

        self.wall_seconds += (
            time.perf_counter() - start
        )

        return np.stack(results)