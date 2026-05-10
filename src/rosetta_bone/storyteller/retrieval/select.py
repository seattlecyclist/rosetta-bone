"""Per-stimulus chunk selection: top-1 chunk from each pillar."""

from __future__ import annotations

from pathlib import Path

from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.logging import get_logger
from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.retrieval.embed import Embedder
from rosetta_bone.storyteller.retrieval.index import PillarIndex

_log = get_logger(__name__)


def _load_chunks(chunks_dir: Path, pillar: Pillar) -> list[Chunk]:
    rows = list(iter_jsonl(chunks_dir / f"{pillar.value}.jsonl"))
    return [Chunk.model_validate(r) for r in rows]


def build_indexes(
    embedder: Embedder,
    *,
    chunks_dir: Path,
    embeddings_dir: Path,
) -> dict[Pillar, tuple[PillarIndex, dict[str, Chunk]]]:
    """Build (or load if cached) FAISS index per pillar.

    Returns map from Pillar to (index, id-to-chunk map) for direct lookup
    after a query.
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    out: dict[Pillar, tuple[PillarIndex, dict[str, Chunk]]] = {}
    for pillar in Pillar:
        chunks = _load_chunks(chunks_dir, pillar)
        if not chunks:
            _log.warning("pillar_empty", pillar=pillar.value)
            continue
        idx_path = embeddings_dir / f"{pillar.value}.faiss"
        ids_path = idx_path.with_suffix(".ids.json")
        id_to_chunk = {c.id: c for c in chunks}
        if idx_path.exists() and ids_path.exists():
            idx = PillarIndex.load(idx_path)
        else:
            vecs = embedder.embed([c.text for c in chunks])
            idx = PillarIndex.build(vecs, [c.id for c in chunks])
            idx.save(idx_path)
        out[pillar] = (idx, id_to_chunk)
    return out


def select_chunks(
    stimulus: str,
    indexes: dict[Pillar, tuple[PillarIndex, dict[str, Chunk]]],
    embedder: Embedder,
    *,
    similarity_threshold: float = 0.25,
) -> dict[Pillar, Chunk]:
    qvec = embedder.embed([stimulus])[0]
    out: dict[Pillar, Chunk] = {}
    for pillar, (idx, id_to_chunk) in indexes.items():
        sims, hits = idx.query(qvec, top_k=1)
        if sims[0] < similarity_threshold:
            _log.warning(
                "low_similarity_match",
                pillar=pillar.value,
                stimulus=stimulus,
                sim=sims[0],
                threshold=similarity_threshold,
            )
        out[pillar] = id_to_chunk[hits[0]]
    return out
