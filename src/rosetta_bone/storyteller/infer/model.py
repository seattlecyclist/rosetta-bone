"""Lazy-cached base+adapter loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_cache: dict[str, tuple[Any, Any]] = {}


def load(base_model: str, adapter_dir: Path | None = None) -> tuple[Any, Any]:
    key = f"{base_model}::{adapter_dir}"
    if key in _cache:
        return _cache[key]
    from mlx_lm import load as mlx_load
    if adapter_dir is not None:
        model, tokenizer = mlx_load(base_model, adapter_path=str(adapter_dir))
    else:
        model, tokenizer = mlx_load(base_model)
    _cache[key] = (model, tokenizer)
    return model, tokenizer
