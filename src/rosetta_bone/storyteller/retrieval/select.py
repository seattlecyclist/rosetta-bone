"""Per-stimulus chunk selection: top-1 chunk from each pillar.

When the caller passes `science_modality`, the science pillar's
candidate set is filtered to chunks whose `metadata.modality` matches
before picking the cosine top-1. The filter looks `MODALITY_POOL` deep
into FAISS results and falls back to the unfiltered top-1 if none
match — that fallback shouldn't fire for the v10 corpus (17 hearing /
17 smell / 16 other across 1.5K chunks) but is logged when it does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.logging import get_logger
from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.retrieval.embed import Embedder
from rosetta_bone.storyteller.retrieval.index import PillarIndex

_log = get_logger(__name__)

Modality = Literal["smell", "hearing"]

# How many cosine-top results to consider when filtering by modality.
# Sized so the matching-modality set is ~always non-empty: with
# ~1500 science chunks evenly split across smell/hearing/other, a
# pool of 50 contains ~17 of each modality on average.
MODALITY_POOL = 50


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
    science_modality: Modality | None = None,
) -> dict[Pillar, Chunk]:
    qvec = embedder.embed([stimulus])[0]
    out: dict[Pillar, Chunk] = {}
    for pillar, (idx, id_to_chunk) in indexes.items():
        if pillar == Pillar.SCIENCE and science_modality is not None:
            sims, hits = idx.query(qvec, top_k=MODALITY_POOL)
            chosen_id, chosen_sim = None, None
            for s, h in zip(sims, hits, strict=True):
                if id_to_chunk[h].metadata.get("modality") == science_modality:
                    chosen_id, chosen_sim = h, s
                    break
            if chosen_id is None:
                _log.warning(
                    "modality_filter_empty",
                    stimulus=stimulus,
                    modality=science_modality,
                    pool=MODALITY_POOL,
                )
                chosen_id, chosen_sim = hits[0], sims[0]
        else:
            sims, hits = idx.query(qvec, top_k=1)
            chosen_id, chosen_sim = hits[0], sims[0]
        if chosen_sim < similarity_threshold:
            _log.warning(
                "low_similarity_match",
                pillar=pillar.value,
                stimulus=stimulus,
                sim=chosen_sim,
                threshold=similarity_threshold,
            )
        out[pillar] = id_to_chunk[chosen_id]
    return out
