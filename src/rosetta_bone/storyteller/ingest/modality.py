"""Sensory-modality tagging for the science pillar.

A chunk's modality is set from the *paper title* it came from. Title
match is coarse but stable: "olfact*", "scent", "vomeronasal", "nasal"
=> smell; "hear*", "audit*", "cochlea", "BAER", "noise", "deaf*",
"sound", "acoust*", "pinna", "presbycus*", "ultrasonic" => hearing.
Papers whose titles match neither (clinical / comparative work that
mentions dogs but isn't sensory) get None and behave like the v9
unfiltered pool.

Used both by:
  - `ingest-inspect` to summarize the smell/hearing balance of raw_dir
  - `chunk_pillar` to stamp modality into Chunk metadata so retrieval
    can filter by it
"""

from __future__ import annotations

import re
from typing import Literal

Modality = Literal["smell", "hearing"]

_PATTERNS: list[tuple[Modality, re.Pattern[str]]] = [
    ("smell", re.compile(r"olfact|scent|nasal|vomeronasal|odou?r|sniff", re.I)),
    ("hearing", re.compile(
        r"hear|audit|cochlea|baer|deaf|noise|sound|acoust|pinna|ultrasonic|presbycus",
        re.I,
    )),
]


def classify_title(title: str) -> Modality | None:
    """Return the first matching modality bucket, or None."""
    for name, pat in _PATTERNS:
        if pat.search(title):
            return name
    return None
