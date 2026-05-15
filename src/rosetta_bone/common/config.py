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
class RemoteTrain:
    base_model: str
    image: str
    gpu_type: str
    bucket: str
    endpoint_url: str
    pod_timeout_seconds: int


@dataclass(frozen=True)
class Train:
    base_model: str
    rank: int
    alpha: float
    iters: int
    batch_size: int
    learning_rate: float
    target_modules: tuple[str, ...]
    remote: RemoteTrain | None = None


@dataclass(frozen=True)
class Infer:
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    retrieval: Retrieval
    sft: Sft
    train: Train
    infer: Infer


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    train_raw = {**raw["train"]}
    remote_raw = train_raw.pop("remote", None)
    remote = RemoteTrain(**remote_raw) if remote_raw else None
    return Config(
        paths=Paths(**{k: Path(v) for k, v in raw["paths"].items()}),
        retrieval=Retrieval(**raw["retrieval"]),
        sft=Sft(**raw["sft"]),
        train=Train(
            **{**train_raw, "target_modules": tuple(train_raw["target_modules"])},
            remote=remote,
        ),
        infer=Infer(**raw["infer"]),
    )
