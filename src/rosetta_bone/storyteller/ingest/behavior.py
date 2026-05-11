"""HuggingFace 'behavior' pillar loader.

Uses pawgaze/pawgaze. The dataset is a visual-Q&A benchmark whose
schema is:

  idx, dataset, scene_name, question_category, question,
  ground_truth (letter A|B|C|D), options (list of 'A. ...' strings).

The signal we want for the behavior pillar lives in `question` (which
narrates dog stimulus context) plus the option text matching
`ground_truth` (which narrates the correct stimulus-reaction
interpretation). We extract both into a single behavior text per row.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rosetta_bone.common.jsonl import write_all
from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)

_DATASET = "pawgaze/pawgaze"


def _pick_correct_option(options: list[str] | None, ground_truth: str | None) -> str | None:
    """Return the option text matching `ground_truth` letter, stripped of its prefix.

    `options` look like ["A. text", "B. text", ...]; ground_truth is one of
    A|B|C|D. Returns the text after "A. " (or returns None on mismatch).
    """
    if not options or not ground_truth:
        return None
    prefix = f"{ground_truth}."
    for opt in options:
        if isinstance(opt, str) and opt.startswith(prefix):
            return opt[len(prefix):].lstrip()
    return None


def extract_text_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project pawgaze rows into {source, text, metadata} records.

    text = "{question}\\n\\n{correct_option_text}". Rows where the
    correct option can't be resolved are skipped.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        question = (r.get("question") or "").strip()
        correct = _pick_correct_option(r.get("options"), r.get("ground_truth"))
        if not question or not correct:
            continue
        text = f"{question}\n\n{correct}"
        rid = r.get("idx", len(out))
        meta = {
            "scene_name": r.get("scene_name"),
            "question_category": r.get("question_category"),
            "ground_truth": r.get("ground_truth"),
        }
        out.append({
            "source": f"{_DATASET}:{rid}",
            "text": text,
            "metadata": meta,
        })
    return out


def fetch_behavior(raw_dir: Path, *, limit: int = 1000) -> Path:
    """Load HF dataset and serialize to raw_dir/pawgaze.jsonl.

    The dataset only publishes a `test` split (no `train` split), so we
    load that one.
    """
    from datasets import load_dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "pawgaze.jsonl"
    if out_path.exists():
        _log.info("behavior_skip_existing", path=str(out_path))
        return out_path

    _log.info("behavior_fetch", dataset=_DATASET, limit=limit)
    ds = load_dataset(_DATASET, split=f"test[:{limit}]")
    rows = extract_text_rows([dict(r) for r in ds])
    write_all(out_path, rows)
    _log.info("behavior_done", n_rows=len(rows), out=str(out_path))
    return out_path
