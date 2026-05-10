"""Merge raw batch results → train.jsonl + valid.jsonl (mlx-lm chat format).

- Validate that each assistant text is JSON with {instruction, story}.
- Dedup by SHA-1 of normalized instruction text.
- Split 90/10 (configurable, seeded) into train/valid.
- Compute a 5-gram grounding stat against the science chunk for the merge
  log; warn if average ratio falls below 30%.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rosetta_bone.common.jsonl import iter_jsonl, write_all
from rosetta_bone.common.logging import get_logger
from rosetta_bone.storyteller.sft.cost import Usage, sum_usage

_log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED_KEYS = {"instruction", "story"}


@dataclass
class MergeStats:
    kept: int
    deduped: int
    dropped_invalid: int


def parse_assistant_json(text: str) -> dict | None:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not _REQUIRED_KEYS.issubset(obj):
        return None
    return {k: obj[k] for k in _REQUIRED_KEYS}


def _hash_instr(s: str) -> str:
    norm = re.sub(r"\s+", " ", s.strip().lower())
    return hashlib.sha1(norm.encode()).hexdigest()


def grounding_5gram_ratio(story: str, science_text: str) -> float:
    """Fraction of 5-grams in story that also appear in science_text."""
    def grams(s: str) -> set[tuple[str, ...]]:
        toks = re.findall(r"\w+", s.lower())
        return {tuple(toks[i : i + 5]) for i in range(len(toks) - 4)}

    sg = grams(story)
    if not sg:
        return 0.0
    cg = grams(science_text)
    return len(sg & cg) / len(sg)


def merge(
    *,
    batches_dir: Path,
    train_path: Path,
    valid_path: Path,
    valid_fraction: float = 0.1,
    seed: int = 1337,
) -> MergeStats:
    pairs: dict[str, dict] = {}
    dropped = 0
    deduped = 0
    usages: list[Usage] = []
    for batch_file in sorted(batches_dir.glob("*.jsonl")):
        for row in iter_jsonl(batch_file):
            if row.get("type") != "succeeded":
                dropped += 1
                continue
            if "usage" in row:
                usages.append(Usage(**row["usage"]))
            parsed = parse_assistant_json(row.get("text", ""))
            if parsed is None:
                dropped += 1
                continue
            key = _hash_instr(parsed["instruction"])
            if key in pairs:
                deduped += 1
                continue
            pairs[key] = parsed

    rows = list(pairs.values())
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_fraction)) if rows else 0
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    def to_chat(p: dict) -> dict:
        return {"messages": [
            {"role": "user", "content": p["instruction"]},
            {"role": "assistant", "content": p["story"]},
        ]}

    write_all(train_path, [to_chat(p) for p in train_rows])
    write_all(valid_path, [to_chat(p) for p in valid_rows])

    stats = MergeStats(kept=len(rows), deduped=deduped, dropped_invalid=dropped)
    totals = sum_usage(usages) if usages else None
    _log.info(
        "merge_done",
        **stats.__dict__,
        n_train=len(train_rows),
        n_valid=len(valid_rows),
        total_input_tokens=totals.input_tokens if totals else 0,
        total_output_tokens=totals.output_tokens if totals else 0,
        total_cache_read_input_tokens=totals.cache_read_input_tokens if totals else 0,
    )
    return stats
