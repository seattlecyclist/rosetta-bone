"""sentence-transformers wrapper that emits L2-normalized vectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        # sentence-transformers >=3 renamed the method; fall back for older.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None,
        ) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
