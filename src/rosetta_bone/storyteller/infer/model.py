"""Lazy-cached base+adapter loader.

The `adapter_dir` accepted here is the *versioned root* configured in
config/default.toml (e.g., `data/adapters/llama31-8b-storyteller-v1`).
Actual adapter weights live one level deeper, under a per-train-run
timestamp directory, with a `latest` symlink pointing at the most
recent run. Inference resolves through `latest`.

Back-compat: if the root itself contains adapter weights (older
layouts, or hand-extracted adapters), it's used directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_cache: dict[str, tuple[Any, Any]] = {}


def _resolve_adapter_dir(adapter_dir: Path | None) -> Path | None:
    """Map the configured root to the actual adapter weights directory.

    Resolution order:
      1. `<root>/latest` symlink or directory  → return it.
      2. `<root>` itself contains adapter weights (legacy)  → return it.
      3. Otherwise  → return None (no adapter; load base model only).
    """
    if adapter_dir is None:
        return None
    latest = adapter_dir / "latest"
    if latest.exists():
        return latest
    if adapter_dir.exists() and any(adapter_dir.glob("*.safetensors")):
        return adapter_dir
    return None


def load(base_model: str, adapter_dir: Path | None = None) -> tuple[Any, Any]:
    resolved = _resolve_adapter_dir(adapter_dir)
    key = f"{base_model}::{resolved}"
    if key in _cache:
        return _cache[key]
    from mlx_lm import load as mlx_load
    if resolved is not None:
        model, tokenizer = mlx_load(base_model, adapter_path=str(resolved))
    else:
        model, tokenizer = mlx_load(base_model)
    _cache[key] = (model, tokenizer)
    return model, tokenizer
