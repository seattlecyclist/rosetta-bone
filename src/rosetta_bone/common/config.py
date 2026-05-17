"""TOML config loader. Single source of truth for paths + hyperparameters."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    raw_dir: Path
    chunks_dir: Path
    embeddings_dir: Path
    sft_dir: Path
    adapter_dir: Path


@dataclass(frozen=True)
class Retrieval:
    embedding_model: str
    similarity_threshold: float


@dataclass(frozen=True)
class Sft:
    model: str
    max_requests_per_run: int
    requests_per_minute: int
    batch_size_max: int


@dataclass(frozen=True)
class Train:
    base_model: str
    rank: int
    alpha: float
    iters: int
    batch_size: int
    learning_rate: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class Infer:
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float


@dataclass(frozen=True)
class Persona:
    module: str = "rosetta_bone.storyteller.sft.persona"


@dataclass(frozen=True)
class Config:
    paths: Paths
    retrieval: Retrieval
    sft: Sft
    train: Train
    infer: Infer
    persona: Persona


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    return Config(
        paths=Paths(**{k: Path(v) for k, v in raw["paths"].items()}),
        retrieval=Retrieval(**raw["retrieval"]),
        sft=Sft(**raw["sft"]),
        train=Train(
            **{**raw["train"], "target_modules": tuple(raw["train"]["target_modules"])}
        ),
        infer=Infer(**raw["infer"]),
        persona=Persona(**raw.get("persona", {})),
    )
