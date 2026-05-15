"""Load and expand stimuli.yaml.

Each stimulus declares one user-facing `prompt` plus one or more
`embed_queries` — retrieval-driving phrasings that vary the FAISS
result per angle. For N angles and M `variations_per_query`, the
stimulus produces N*M total SFT-pair requests, each with its own
retrieved chunks AND its own `Angle:` hint in the user block.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Form = Literal["diary", "vignette", "short_story"]
Modality = Literal["smell", "hearing"]


class Stimulus(BaseModel):
    prompt: str
    embed_queries: list[str]
    variations_per_query: int = Field(ge=1)
    form: Form
    # Optional sensory hint that constrains science-pillar retrieval to
    # chunks whose source paper title matches this modality. Unset =
    # search the whole pillar (v9 behaviour). Set on stimuli where the
    # narrative arc hangs on a specific sense, like the v10 auditory
    # additions.
    modality: Modality | None = None

    @field_validator("embed_queries")
    @classmethod
    def _at_least_one_query(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("embed_queries must have at least one entry")
        if any(not q.strip() for q in v):
            raise ValueError("embed_queries entries must be non-empty strings")
        return v


def load_stimuli(path: Path) -> list[Stimulus]:
    raw = yaml.safe_load(path.read_text())
    return [Stimulus.model_validate(r) for r in raw]


def expand(
    stimuli: list[Stimulus],
) -> Iterator[tuple[str, str, int, Form, Modality | None]]:
    """Yield (stimulus, embed_query, variation_idx, form, modality) 5-tuples.

    Variation index is per-(stimulus, embed_query) — resets at the start
    of each angle. `modality` is the stimulus-level sensory hint (or
    None) and propagates to retrieval so the science-pillar candidate
    set can be filtered before cosine ranking.
    """
    for s in stimuli:
        for query in s.embed_queries:
            for v in range(s.variations_per_query):
                yield s.prompt, query, v, s.form, s.modality
