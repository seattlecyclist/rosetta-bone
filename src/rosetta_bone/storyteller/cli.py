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


@app.command("embed")
def embed_cmd(
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    """Build FAISS indexes for all pillars."""
    from rosetta_bone.storyteller.retrieval.embed import Embedder
    from rosetta_bone.storyteller.retrieval.select import build_indexes

    cfg = load_config(config_path)
    embedder = Embedder(cfg.retrieval.embedding_model)
    build_indexes(
        embedder,
        chunks_dir=cfg.paths.chunks_dir,
        embeddings_dir=cfg.paths.embeddings_dir,
    )


sft_app = typer.Typer(help="SFT-pair generation pipeline", no_args_is_help=True)
app.add_typer(sft_app, name="sft")


@sft_app.command("generate")
def sft_generate(
    count: int = typer.Option(..., help="Total SFT pairs to generate"),
    phase: str = typer.Option("pilot", help="Phase tag: pilot | full"),
    max_requests: int | None = typer.Option(None, "--max-requests"),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    import os
    from itertools import islice

    from anthropic import Anthropic
    from dotenv import load_dotenv

    from rosetta_bone.storyteller.retrieval.embed import Embedder
    from rosetta_bone.storyteller.retrieval.select import (
        build_indexes,
        select_chunks,
    )
    from rosetta_bone.storyteller.sft.generate import (
        enforce_request_cap,
        plan_batch,
        submit_batch,
    )
    from rosetta_bone.storyteller.sft.stimuli import expand, load_stimuli

    load_dotenv()
    cfg = load_config(config_path)
    cap = max_requests if max_requests is not None else cfg.sft.max_requests_per_run
    try:
        enforce_request_cap(count=count, cap=cap)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    stimuli = load_stimuli(Path("config/stimuli.yaml"))
    triples = list(islice(expand(stimuli), count))

    embedder = Embedder(cfg.retrieval.embedding_model)
    indexes = build_indexes(
        embedder,
        chunks_dir=cfg.paths.chunks_dir,
        embeddings_dir=cfg.paths.embeddings_dir,
    )

    def selector(stim: str):
        return select_chunks(
            stim, indexes, embedder,
            similarity_threshold=cfg.retrieval.similarity_threshold,
        )

    plan = plan_batch(triples, select_fn=selector, model=cfg.sft.model, phase=phase)
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = Anthropic(api_key=api_key)
    bid = submit_batch(plan, client=client,
                       manifest_path=cfg.paths.sft_dir / "manifest.jsonl")
    typer.echo(f"Submitted batch {bid} with {len(plan.requests)} requests.")


@sft_app.command("poll")
def sft_poll(
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    import os

    from anthropic import Anthropic
    from dotenv import load_dotenv

    from rosetta_bone.storyteller.sft.poll import poll_once

    load_dotenv()
    cfg = load_config(config_path)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pending = poll_once(
        client=client,
        manifest_path=cfg.paths.sft_dir / "manifest.jsonl",
        out_dir=cfg.paths.sft_dir / "batches",
    )
    if pending:
        typer.echo(f"{len(pending)} batch(es) still in progress: " +
                   ", ".join(f"{b.batch_id}={b.status}" for b in pending))
    else:
        typer.echo("All batches downloaded.")


@sft_app.command("merge")
def sft_merge(
    valid_fraction: float = typer.Option(0.1),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    from rosetta_bone.storyteller.sft.merge import merge

    cfg = load_config(config_path)
    stats = merge(
        batches_dir=cfg.paths.sft_dir / "batches",
        train_path=cfg.paths.sft_dir / "train.jsonl",
        valid_path=cfg.paths.sft_dir / "valid.jsonl",
        valid_fraction=valid_fraction,
    )
    typer.echo(f"Kept {stats.kept}, deduped {stats.deduped}, dropped {stats.dropped_invalid}.")


@app.command("train")
def train_cmd(
    iters: int = typer.Option(1000, help="Training iterations"),
    batch_size: int | None = typer.Option(
        None, "--batch-size",
        help="Override config batch_size. If unset, uses config and auto-clamps "
             "to the train-set size for small pilots.",
    ),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    from rosetta_bone.common.jsonl import iter_jsonl
    from rosetta_bone.storyteller.train.lora import train

    cfg = load_config(config_path)
    train_path = cfg.paths.sft_dir / "train.jsonl"

    # mlx-lm requires batch_size <= n_train_rows. For small pilots
    # (10-50 SFT pairs), the default batch_size of 4 from default.toml
    # may exceed the merged train set. Auto-clamp with a warning.
    requested = batch_size if batch_size is not None else cfg.train.batch_size
    n_train = sum(1 for _ in iter_jsonl(train_path))
    if n_train == 0:
        typer.echo(
            f"No training data at {train_path}. Did sft merge run?",
            err=True,
        )
        raise typer.Exit(code=2)
    effective = min(requested, n_train)
    if effective < requested:
        typer.echo(
            f"Clamped batch_size {requested} -> {effective} "
            f"(train.jsonl has only {n_train} rows).",
        )

    res = train(
        base_model=cfg.train.base_model,
        train_data=train_path,
        valid_data=cfg.paths.sft_dir / "valid.jsonl",
        adapter_dir=cfg.paths.adapter_dir,
        rank=cfg.train.rank, alpha=cfg.train.alpha,
        iters=iters, batch_size=effective,
        learning_rate=cfg.train.learning_rate,
    )
    if res.returncode != 0:
        typer.echo(res.stderr, err=True)
        raise typer.Exit(code=res.returncode)
    typer.echo("Training complete.")


@app.command("generate")
def generate_cmd(
    stimulus: str = typer.Argument(..., help="The stimulus prompt, e.g., 'a trip to the vet'"),
    form: str = typer.Option("diary", help="diary | vignette | short_story"),
    max_tokens: int | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    top_p: float | None = typer.Option(None),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    from rosetta_bone.storyteller.infer.generate import generate

    text = generate(
        stimulus, form=form, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, config_path=config_path,
    )
    typer.echo(text)
