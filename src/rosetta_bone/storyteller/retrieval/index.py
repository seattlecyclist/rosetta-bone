"""Per-pillar FAISS IndexFlatIP with separate JSON id list."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class PillarIndex:
    index: faiss.Index
    ids: list[str]

    @classmethod
    def build(cls, vecs: np.ndarray, ids: list[str]) -> PillarIndex:
        assert vecs.shape[0] == len(ids)
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        return cls(idx, ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        path.with_suffix(".ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, path: Path, ids: list[str] | None = None) -> PillarIndex:
        index = faiss.read_index(str(path))
        if ids is None:
            ids = json.loads(path.with_suffix(".ids.json").read_text())
        return cls(index, ids)

    def query(self, vec: np.ndarray, *, top_k: int = 1) -> tuple[list[float], list[str]]:
        v = vec.reshape(1, -1).astype(np.float32)
        sims, hits = self.index.search(v, top_k)
        return [float(s) for s in sims[0]], [self.ids[h] for h in hits[0]]
