"""Ingest pipeline orchestrator.

Per-pillar entry points each consume `raw_dir` (PDFs, text files, or a
JSONL fallback for behavior) and produce a uniform chunks JSONL file.
"""

from __future__ import annotations

import json
from pathlib import Path

from rosetta_bone.common.chunking import chunk_text
from rosetta_bone.common.jsonl import iter_jsonl, write_all
from rosetta_bone.common.logging import get_logger
from rosetta_bone.common.pdf import pdf_to_text
from rosetta_bone.common.types import Pillar
from rosetta_bone.storyteller.ingest.modality import classify_title

_log = get_logger(__name__)


def _science_metadata(pdf_path: Path) -> dict:
    """Pull title + modality from the {pmcid}.json sidecar if present.

    Modality is set when the title matches the smell/hearing patterns
    in `ingest.modality`; otherwise omitted, which leaves the chunk in
    the unfiltered fallback pool at retrieval time.
    """
    sidecar = pdf_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return {}
    title = meta.get("title", "")
    out: dict = {"title": title, "pubYear": meta.get("pubYear")}
    modality = classify_title(title)
    if modality is not None:
        out["modality"] = modality
    return out


def _iter_text_sources(raw_dir: Path, pillar: Pillar):
    """Yield (source_id, text, metadata) tuples for the pillar's raw files."""
    if pillar in (Pillar.STYLE,):
        for p in sorted(raw_dir.glob("*.txt")):
            yield p.stem, p.read_text(), {}
    elif pillar == Pillar.SCIENCE:
        for p in sorted(raw_dir.glob("*.pdf")):
            try:
                t = pdf_to_text(p)
            except Exception as e:
                _log.warning("pdf_skip", path=str(p), error=str(e))
                continue
            if t.strip():
                yield p.stem, t, _science_metadata(p)
    elif pillar == Pillar.BEHAVIOR:
        for p in sorted(raw_dir.glob("*.jsonl")):
            for row in iter_jsonl(p):
                yield row["source"], row["text"], row.get("metadata", {})


def chunk_pillar(
    *,
    raw_dir: Path,
    pillar: Pillar,
    out_path: Path,
    target_tokens: int = 600,
    overlap: int = 80,
) -> Path:
    rows = []
    for source_id, text, meta in _iter_text_sources(raw_dir, pillar):
        for c in chunk_text(
            text,
            source_id=source_id,
            pillar=pillar,
            metadata=meta,
            target_tokens=target_tokens,
            overlap=overlap,
        ):
            rows.append(c.model_dump(mode="json"))
    write_all(out_path, rows)
    _log.info("chunked_pillar", pillar=pillar.value, n_chunks=len(rows),
              out=str(out_path))
    return out_path
