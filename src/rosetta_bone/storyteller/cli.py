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
    wait: bool = typer.Option(
        False, "--wait/--no-wait",
        help="Block until all submitted batches are downloaded.",
    ),
    interval: int = typer.Option(
        30, "--interval",
        help="Seconds between polls when --wait is set.",
    ),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    import os
    import time

    from anthropic import Anthropic
    from dotenv import load_dotenv

    from rosetta_bone.storyteller.sft.poll import poll_once

    load_dotenv()
    cfg = load_config(config_path)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    manifest = cfg.paths.sft_dir / "manifest.jsonl"
    out_dir = cfg.paths.sft_dir / "batches"

    while True:
        pending = poll_once(client=client, manifest_path=manifest, out_dir=out_dir)
        if not pending:
            typer.echo("All batches downloaded.")
            return
        msg = (
            f"{len(pending)} batch(es) still in progress: "
            + ", ".join(f"{b.batch_id}={b.status}" for b in pending)
        )
        if not wait:
            typer.echo(msg)
            return
        typer.echo(f"{msg} -- sleeping {interval}s...")
        time.sleep(interval)


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
    import hashlib
    import json
    import time
    from datetime import UTC, datetime

    from rosetta_bone.common.jsonl import iter_jsonl
    from rosetta_bone.storyteller.train.lora import train

    cfg = load_config(config_path)
    train_path = cfg.paths.sft_dir / "train.jsonl"
    valid_path = cfg.paths.sft_dir / "valid.jsonl"

    requested = batch_size if batch_size is not None else cfg.train.batch_size
    n_train = sum(1 for _ in iter_jsonl(train_path))
    n_valid = sum(1 for _ in iter_jsonl(valid_path))
    if n_train == 0:
        typer.echo(f"No training data at {train_path}. Did sft merge run?", err=True)
        raise typer.Exit(code=2)
    if n_valid == 0:
        typer.echo(f"No validation data at {valid_path}. Did sft merge run?", err=True)
        raise typer.Exit(code=2)
    effective = min(requested, n_train, n_valid)
    if effective < requested:
        typer.echo(
            f"Clamped batch_size {requested} -> {effective} "
            f"(train.jsonl has {n_train} rows, valid.jsonl has {n_valid}).",
        )

    # Versioned adapter directory: data/adapters/.../{timestamp}/ plus a
    # 'latest' symlink that inference reads through. Each train run
    # leaves the previous run intact for comparison and rollback.
    adapter_root = cfg.paths.adapter_dir
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    versioned_dir = adapter_root / timestamp
    adapter_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    res = train(
        base_model=cfg.train.base_model,
        train_data=train_path,
        valid_data=valid_path,
        adapter_dir=versioned_dir,
        rank=cfg.train.rank, alpha=cfg.train.alpha,
        iters=iters, batch_size=effective,
        learning_rate=cfg.train.learning_rate,
    )
    duration_s = time.monotonic() - started

    if res.returncode != 0:
        # mlx-lm's own stderr already streamed live; nothing more to print.
        raise typer.Exit(code=res.returncode)

    # Sidecar metadata so a future operator can answer "what data and
    # hyperparams produced this adapter?" without grepping shell history.
    def _sha1(p: Path) -> str:
        h = hashlib.sha1()
        with p.open("rb") as f:
            for buf in iter(lambda: f.read(65536), b""):
                h.update(buf)
        return h.hexdigest()

    try:
        import mlx_lm
        mlx_lm_version = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        mlx_lm_version = "unknown"

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": cfg.train.base_model,
        "iters": iters,
        "batch_size_requested": requested,
        "batch_size_effective": effective,
        "learning_rate": cfg.train.learning_rate,
        "rank": cfg.train.rank,
        "alpha": cfg.train.alpha,
        "train_rows": n_train,
        "valid_rows": n_valid,
        "train_sha1": _sha1(train_path),
        "valid_sha1": _sha1(valid_path),
        "mlx_lm_version": mlx_lm_version,
        "duration_seconds": round(duration_s, 2),
    }
    (versioned_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Atomically swap the 'latest' symlink. Use a relative target so the
    # link survives if the parent dir is moved.
    latest = adapter_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(timestamp)

    typer.echo(
        f"Training complete in {duration_s:.1f}s. "
        f"Adapter at {versioned_dir} (latest -> {timestamp}).",
    )


def _resolve_adapter_arg(arg: str, adapter_root: Path) -> Path:
    """Resolve an --adapter argument to a concrete path.

    Accepts either a bare timestamp directory name (looked up under
    cfg.paths.adapter_dir) or an absolute/relative path that already
    points at a versioned adapter directory.
    """
    p = Path(arg)
    if p.is_absolute() or "/" in arg:
        return p
    return adapter_root / arg


@app.command("generate")
def generate_cmd(
    stimulus: str = typer.Argument(..., help="The stimulus prompt, e.g., 'a trip to the vet'"),
    form: str = typer.Option("diary", help="diary | vignette | short_story"),
    max_tokens: int | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    top_p: float | None = typer.Option(None),
    adapter: str | None = typer.Option(
        None, "--adapter",
        help="Override which trained adapter to use. Pass a timestamp like "
             "'20260512T015730Z' (resolved under cfg.paths.adapter_dir) or a "
             "full path to a timestamped adapter dir. Defaults to 'latest'.",
    ),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    from rosetta_bone.storyteller.infer.generate import generate

    cfg = load_config(config_path)
    adapter_override = (
        _resolve_adapter_arg(adapter, cfg.paths.adapter_dir) if adapter else None
    )

    text = generate(
        stimulus, form=form, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        adapter_override=adapter_override,
        config_path=config_path,
    )
    typer.echo(text)


@app.command("eval")
def eval_cmd(
    adapter: str | None = typer.Option(
        None, "--adapter",
        help="Adapter timestamp or full path. Defaults to 'latest'.",
    ),
    eval_set: Path = typer.Option(
        Path("config/eval_prompts.yaml"), "--eval-set",
        help="YAML file listing the prompts to evaluate.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-run even if eval results already exist for this prompt set.",
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    """Run a frozen eval prompt set against one adapter; save results next to it."""
    from rosetta_bone.storyteller.eval_suite import load_prompts, run_eval

    cfg = load_config(config_path)
    if adapter:
        adapter_dir = _resolve_adapter_arg(adapter, cfg.paths.adapter_dir)
    else:
        latest = cfg.paths.adapter_dir / "latest"
        if not latest.exists():
            typer.echo(
                f"No 'latest' symlink at {latest}. Train at least once first.",
                err=True,
            )
            raise typer.Exit(2)
        adapter_dir = latest.resolve()

    if not adapter_dir.exists():
        typer.echo(f"Adapter dir does not exist: {adapter_dir}", err=True)
        raise typer.Exit(2)

    prompts = load_prompts(eval_set)
    out_path = run_eval(
        adapter_dir=adapter_dir,
        base_model=cfg.train.base_model,
        prompts=prompts,
        max_tokens=max_tokens,
        force=force,
    )
    typer.echo(f"Eval results: {out_path}")


@app.command("eval-compare")
def eval_compare_cmd(
    a: str = typer.Argument(..., help="First adapter: timestamp or path."),
    b: str = typer.Argument(..., help="Second adapter: timestamp or path."),
    eval_set: Path = typer.Option(
        Path("config/eval_prompts.yaml"), "--eval-set",
        help="Used to compute eval_sha and find the eval-<sha>.json file.",
    ),
    config_path: Path = typer.Option(Path("config/default.toml"), "--config"),
) -> None:
    """Pretty-print two adapters' eval results side-by-side."""
    import json as _json

    from rosetta_bone.storyteller.eval_suite import (
        compare_evals,
        eval_set_sha,
        load_prompts,
    )

    cfg = load_config(config_path)
    a_dir = _resolve_adapter_arg(a, cfg.paths.adapter_dir)
    b_dir = _resolve_adapter_arg(b, cfg.paths.adapter_dir)
    prompts = load_prompts(eval_set)
    sha = eval_set_sha(prompts)
    a_path = a_dir / f"eval-{sha}.json"
    b_path = b_dir / f"eval-{sha}.json"

    missing = [p for p in (a_path, b_path) if not p.exists()]
    if missing:
        for p in missing:
            typer.echo(
                f"Missing eval file: {p}. "
                f"Run 'rosetta-storyteller eval --adapter {p.parent.name}' first.",
                err=True,
            )
        raise typer.Exit(2)

    a_data = _json.loads(a_path.read_text())
    b_data = _json.loads(b_path.read_text())
    typer.echo(compare_evals(a_data, b_data))
