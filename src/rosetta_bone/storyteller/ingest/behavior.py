"""HuggingFace 'behavior' pillar loader.

Uses pawgaze/pawgaze if available; if the dataset is unreachable, raise
informatively so the user can drop a JSONL fallback into raw_dir.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rosetta_bone.common.jsonl import write_all
from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)

_DATASET = "pawgaze/pawgaze"


def extract_text_rows(
    rows: Iterable[dict[str, Any]],
    *,
    text_field: str = "description",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        t = (r.get(text_field) or "").strip()
        if not t:
            continue
        rid = r.get("id", (out and out[-1]["metadata"].get("id", 0) + 1) or 0)
        meta = {k: v for k, v in r.items() if k != text_field}
        out.append({
            "source": f"{_DATASET}:{rid}",
            "text": t,
            "metadata": meta,
        })
    return out


def fetch_behavior(raw_dir: Path, *, limit: int = 1000) -> Path:
    """Load HF dataset and serialize to raw_dir/pawgaze.jsonl."""
    from datasets import load_dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "pawgaze.jsonl"
    if out_path.exists():
        _log.info("behavior_skip_existing", path=str(out_path))
        return out_path

    _log.info("behavior_fetch", dataset=_DATASET, limit=limit)
    ds = load_dataset(_DATASET, split=f"train[:{limit}]")
    rows = extract_text_rows([dict(r) for r in ds])
    write_all(out_path, rows)
    return out_path
