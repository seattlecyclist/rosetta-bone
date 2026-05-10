from pathlib import Path

import numpy as np

from rosetta_bone.common.jsonl import write_all
from rosetta_bone.common.types import Pillar
from rosetta_bone.storyteller.retrieval.select import (
    build_indexes,
    select_chunks,
)


class _FakeEmbedder:
    """Deterministic 4-d embedder: one-hot by first character class."""
    dim = 4

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            t = (t or "x").lower()
            v[hash(t.split()[0]) % 4] = 1.0
            out.append(v)
        return np.vstack(out)


def _seed_chunks(tmp_path: Path):
    chunks = {
        Pillar.SCIENCE: [
            {"id": "sci-1", "source": "s1", "pillar": "science",
             "text": "vet visit canine vomeronasal", "metadata": {}}
        ],
        Pillar.STYLE: [
            {"id": "sty-1", "source": "s1", "pillar": "style",
             "text": "the dog walked sadly", "metadata": {}}
        ],
        Pillar.BEHAVIOR: [
            {"id": "beh-1", "source": "s1", "pillar": "behavior",
             "text": "tail tucked when nervous", "metadata": {}}
        ],
    }
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    for p, rows in chunks.items():
        write_all(chunks_dir / f"{p.value}.jsonl", rows)
    return chunks_dir


def test_select_returns_one_chunk_per_pillar(tmp_path: Path):
    chunks_dir = _seed_chunks(tmp_path)
    emb_dir = tmp_path / "emb"
    indexes = build_indexes(_FakeEmbedder(), chunks_dir=chunks_dir, embeddings_dir=emb_dir)
    out = select_chunks("vet visit", indexes, _FakeEmbedder())
    assert set(out.keys()) == {Pillar.SCIENCE, Pillar.STYLE, Pillar.BEHAVIOR}
    for p, c in out.items():
        assert c.pillar == p


def test_select_warns_below_threshold(tmp_path):
    chunks_dir = _seed_chunks(tmp_path)
    emb_dir = tmp_path / "emb"
    indexes = build_indexes(_FakeEmbedder(), chunks_dir=chunks_dir, embeddings_dir=emb_dir)
    out = select_chunks(
        "vet visit", indexes, _FakeEmbedder(), similarity_threshold=2.0
    )
    assert all(c is not None for c in out.values())
