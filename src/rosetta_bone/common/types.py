"""Shared types: Pillar enum and Chunk model."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class Pillar(str, Enum):
    SCIENCE = "science"
    STYLE = "style"
    BEHAVIOR = "behavior"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    pillar: Pillar
    text: str
    metadata: dict = {}
