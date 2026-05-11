"""Subprocess wrapper around `mlx_lm.lora`.

Subprocess (not `from mlx_lm.lora import ...`) because mlx-lm's internal
APIs change between releases; CLI args are the stable contract.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def build_train_argv(
    *,
    base_model: str,
    data_dir: Path,
    adapter_dir: Path,
    rank: int,
    alpha: float,
    iters: int,
    batch_size: int,
    learning_rate: float,
) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm.lora",
        "--train",
        "--model", base_model,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--lora-layers", "16",
        "--num-layers", "16",
    ]


def train(
    *,
    base_model: str,
    train_data: Path,
    valid_data: Path,
    adapter_dir: Path,
    rank: int = 8,
    alpha: float = 16.0,
    iters: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
) -> subprocess.CompletedProcess[str]:
    """Invoke mlx_lm.lora as a subprocess.

    mlx-lm expects a directory containing train.jsonl + valid.jsonl. We
    arrange this by ensuring `train_data.parent == valid_data.parent`.
    """
    data_dir = train_data.parent
    if valid_data.parent != data_dir:
        shutil.copy(valid_data, data_dir / "valid.jsonl")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    argv = build_train_argv(
        base_model=base_model, data_dir=data_dir, adapter_dir=adapter_dir,
        rank=rank, alpha=alpha, iters=iters, batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return subprocess.run(argv, check=False, capture_output=True, text=True)
