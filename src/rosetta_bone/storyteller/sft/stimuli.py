"""Load and expand stimuli.yaml."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Form = Literal["diary", "vignette", "short_story"]


class Stimulus(BaseModel):
    prompt: str
    variations: int = Field(ge=1)
    form: Form


def load_stimuli(path: Path) -> list[Stimulus]:
    raw = yaml.safe_load(path.read_text())
    return [Stimulus.model_validate(r) for r in raw]


def expand(stimuli: list[Stimulus]) -> Iterator[tuple[str, int, Form]]:
    for s in stimuli:
        for i in range(s.variations):
            yield s.prompt, i, s.form
