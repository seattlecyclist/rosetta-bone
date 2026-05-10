"""Typer entry point: rosetta-storyteller <subcommand>.

This file imports lazily inside subcommand bodies so `--help` doesn't
pay the cost of loading mlx, sentence-transformers, etc.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rosetta_bone.common.config import load_config
from rosetta_bone.common.logging import configure_logging
from rosetta_bone.common.types import Pillar

app = typer.Typer(help="Rosetta Bone — Dog-POV Storyteller v1 CLI", no_args_is_help=True)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command("ingest")
def ingest_cmd(
    pillar: Pillar = typer.Option(..., help="Which pillar to fetch"),
    limit: int = typer.Option(10, help="Max items to fetch"),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    """Fetch raw source material into data/raw/{pillar}/."""
    cfg = load_config(config_path)
    pillar_dir = cfg.paths.raw_dir / pillar.value
    pillar_dir.mkdir(parents=True, exist_ok=True)

    from rosetta_bone.common.http import CachedClient

    client = CachedClient(cache_dir=cfg.paths.raw_dir / "_cache")
    if pillar == Pillar.STYLE:
        from rosetta_bone.storyteller.ingest.style import fetch_books

        fetch_books(client, pillar_dir, limit=limit)
    elif pillar == Pillar.SCIENCE:
        from rosetta_bone.storyteller.ingest.science import fetch_papers

        fetch_papers(client, pillar_dir, limit=limit)
    elif pillar == Pillar.BEHAVIOR:
        from rosetta_bone.storyteller.ingest.behavior import fetch_behavior

        fetch_behavior(pillar_dir, limit=limit)


@app.command("chunk")
def chunk_cmd(
    pillar: Pillar | None = typer.Option(None, help="Pillar to chunk; --all for every pillar"),
    all_: bool = typer.Option(False, "--all"),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    """Chunk raw files into data/chunks/{pillar}.jsonl."""
    from rosetta_bone.storyteller.ingest.pipeline import chunk_pillar

    cfg = load_config(config_path)
    pillars = list(Pillar) if all_ else ([pillar] if pillar else [])
    if not pillars:
        raise typer.BadParameter("Pass --pillar or --all.")
    cfg.paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    for p in pillars:
        chunk_pillar(
            raw_dir=cfg.paths.raw_dir / p.value,
            pillar=p,
            out_path=cfg.paths.chunks_dir / f"{p.value}.jsonl",
        )
