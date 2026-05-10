"""Poll Anthropic for submitted batches and download results."""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Any

from rosetta_bone.common.jsonl import append, iter_jsonl, write_all
from rosetta_bone.common.logging import get_logger
from rosetta_bone.storyteller.sft.cost import PRICE_TABLE, Usage, estimate_cost_usd, sum_usage

_log = get_logger(__name__)

PENDING_BATCH = namedtuple("PENDING_BATCH", "batch_id status")


def _last_status_per_batch(manifest_path: Path) -> dict[str, str]:
    """Last-write-wins status map per batch_id."""
    out: dict[str, str] = {}
    for row in iter_jsonl(manifest_path):
        bid = row.get("batch_id")
        if bid:
            out[bid] = row["status"]
    return out


def _model_for_batch(manifest_path: Path, batch_id: str) -> str | None:
    """Find the model the batch was submitted with, by scanning the manifest."""
    for row in iter_jsonl(manifest_path):
        if row.get("batch_id") == batch_id and row.get("model"):
            return row["model"]
    return None


def _cost_summary(rows: list[dict[str, Any]], model: str | None) -> dict[str, Any]:
    """Aggregate token + cost stats across the result rows of one batch."""
    usages = [Usage(**r["usage"]) for r in rows if "usage" in r]
    if not usages:
        return {"n_succeeded": 0}
    totals = sum_usage(usages)
    out: dict[str, Any] = {
        "n_succeeded": len(usages),
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "cache_read_input_tokens": totals.cache_read_input_tokens,
        "cache_creation_input_tokens": totals.cache_creation_input_tokens,
    }
    if model:
        out["model"] = model
        if model in PRICE_TABLE:
            out["estimated_cost_usd"] = estimate_cost_usd(totals, model=model, batch=True)
    return out


def poll_once(
    *,
    client: Any,
    manifest_path: Path,
    out_dir: Path,
) -> list[PENDING_BATCH]:
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = _last_status_per_batch(manifest_path)
    pending: list[PENDING_BATCH] = []
    for bid, status in statuses.items():
        if status == "downloaded":
            continue
        b = client.messages.batches.retrieve(bid)
        ps = b.processing_status
        if ps != "ended":
            _log.info("batch_in_progress", batch_id=bid, status=ps)
            pending.append(PENDING_BATCH(bid, ps))
            continue
        rows: list[dict[str, Any]] = []
        for r in client.messages.batches.results(bid):
            row: dict[str, Any] = {"custom_id": r.custom_id, "type": r.result.type}
            if r.result.type == "succeeded":
                msg = r.result.message
                row["text"] = msg.content[0].text if msg.content else ""
                u = msg.usage
                row["usage"] = {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cache_read_input_tokens": (
                        getattr(u, "cache_read_input_tokens", 0) or 0
                    ),
                    "cache_creation_input_tokens": (
                        getattr(u, "cache_creation_input_tokens", 0) or 0
                    ),
                }
            else:
                row["error"] = json.dumps(getattr(r.result, "error", "unknown"), default=str)
            rows.append(row)
        write_all(out_dir / f"{bid}.jsonl", rows)
        cost = _cost_summary(rows, _model_for_batch(manifest_path, bid))
        append(manifest_path, {"batch_id": bid, "status": "downloaded",
                               "n_results": len(rows), **cost})
        _log.info("batch_downloaded", batch_id=bid, n=len(rows), **cost)
        if cost.get("cache_read_input_tokens", 0) == 0 and cost.get("n_succeeded", 0) > 0:
            _log.warning(
                "no_cache_hits",
                batch_id=bid,
                hint=("system block did not benefit from prompt caching; "
                      "verify persona+contract precede the chunks block."),
            )
    return pending
