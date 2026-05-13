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
        # LoRA on the top 8 transformer blocks. Standard for stylistic
        # fine-tunes; halves trainable-parameter count vs 16 and roughly
        # halves the backward-pass work.
        "--num-layers", "8",
        # Cap sequence length. Our SFT pairs are first-person dog stories
        # of ~300-800 tokens; mlx-lm's default of 2048 wastes memory on
        # right-padding and dominates per-iter latency on 8B-4bit.
        "--max-seq-length", "1024",
        # Note: --grad-checkpoint was previously here but removed after
        # observing peak memory of ~9.7 GB on 32 GB unified memory in
        # the v8 training run (well under budget). Removing it should
        # cut iter time by ~25-30 % and bump peak memory to ~17 GB —
        # still comfortable headroom. Rollback signal: any future
        # OOM during train means add it back.
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
    # Tee mlx-lm's stdout to two places:
    #   1. The parent terminal — so the user sees per-iter progress live
    #   2. `<adapter_dir>/train.log` — so `train-inspect` can parse the
    #      training history after the fact (validation-loss series,
    #      throughput, peak memory, overfit verdict)
    # Capturing into a buffer is NOT used (would hide a multi-hour run
    # behind a silent block).
    log_path = adapter_dir / "train.log"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
    return subprocess.CompletedProcess(args=argv, returncode=rc)
