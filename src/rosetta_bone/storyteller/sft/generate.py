"""Plan + submit Anthropic Message Batches for SFT-pair generation.

Per-angle retrieval is cached so all variations sharing the same
embed_query reuse the same chunks (this also maximizes prompt-cache
hits server-side within a stimulus+angle group).

Manifest discipline: every batch is written to data/sft/manifest.jsonl
BEFORE the network call returns. A crash mid-submit leaves the manifest
in sync with what was actually sent.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosetta_bone.common.jsonl import append
from rosetta_bone.common.logging import get_logger
from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.sft.prompt_builder import build_messages

_log = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Anthropic Batches require custom_id to match ^[a-zA-Z0-9_-]{1,64}$ —
# no colons, no other punctuation. Slug separator is `__`. With the
# four-component custom_id `{phase}__{slug}__a{angle_idx}__v{var_idx}`
# (separators alone = 6 chars), the slug is clamped at 30 to leave
# headroom for a typical phase tag plus two-digit indices.
_CUSTOM_ID_SEP = "__"
_SLUG_MAX_LEN = 30


def _slug(s: str) -> str:
    cleaned = _SLUG_RE.sub("-", s.lower()).strip("-")
    return cleaned[:_SLUG_MAX_LEN].rstrip("-")


@dataclass(frozen=True)
class BatchRequest:
    custom_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class BatchPlan:
    requests: list[BatchRequest]
    model: str
    phase: str


def enforce_request_cap(*, count: int, cap: int) -> None:
    if count > cap:
        raise ValueError(
            f"Request count {count} exceeds cap {cap}. "
            f"Override with --max-requests {count} (or higher) if intentional."
        )


def plan_batch(
    triples: Iterable[tuple[str, str, int, str, str | None]],
    *,
    select_fn: Callable[[str, str | None], dict[Pillar, Chunk]],
    model: str,
    phase: str,
) -> BatchPlan:
    """Plan one batch from (stimulus, embed_query, variation, form, modality) tuples.

    Chunks are cached per (embed_query, modality) — modality changes
    the candidate set at retrieval time, so two stimuli that share an
    angle string but differ in modality must not collide in the cache.

    `custom_id` is `{phase}__{slug(stimulus)}__a{angle_idx}__v{var_idx}`
    where `angle_idx` is a small integer assigned in first-seen order
    per stimulus (so different stimuli can reuse `a0`).
    """
    chunk_cache: dict[tuple[str, str | None], dict[Pillar, Chunk]] = {}
    angle_idx_per_stim: dict[str, dict[str, int]] = {}
    requests: list[BatchRequest] = []
    for stimulus, query, variation, form, modality in triples:
        cache_key = (query, modality)
        if cache_key not in chunk_cache:
            chunk_cache[cache_key] = select_fn(query, modality)
        angle_map = angle_idx_per_stim.setdefault(stimulus, {})
        if query not in angle_map:
            angle_map[query] = len(angle_map)
        a_idx = angle_map[query]

        msgs = build_messages(
            chunk_cache[cache_key],
            stimulus=stimulus,
            angle=query,
            form=form,
            variation=variation,
        )
        sep = _CUSTOM_ID_SEP
        requests.append(BatchRequest(
            custom_id=f"{phase}{sep}{_slug(stimulus)}{sep}a{a_idx}{sep}v{variation}",
            messages=msgs,
        ))
    return BatchPlan(requests=requests, model=model, phase=phase)


def _to_anthropic_request(r: BatchRequest, *, model: str, max_tokens: int = 1500) -> dict[str, Any]:
    """Anthropic Batch request shape (Messages-API beta).

    Splits the system message out (Anthropic SDK takes `system` as a
    top-level param, not as a role inside `messages`).
    """
    sys_msg = next((m for m in r.messages if m["role"] == "system"), None)
    rest = [m for m in r.messages if m["role"] != "system"]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": rest,
    }
    if sys_msg is not None:
        params["system"] = [{
            "type": "text",
            "text": sys_msg["content"],
            "cache_control": {"type": "ephemeral"},
        }]
    return {"custom_id": r.custom_id, "params": params}


def submit_batch(
    plan: BatchPlan,
    *,
    client: Any,
    manifest_path: Path,
    max_tokens: int = 1500,
) -> str:
    requests = [_to_anthropic_request(r, model=plan.model, max_tokens=max_tokens)
                for r in plan.requests]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pending_row = {
        "phase": plan.phase,
        "model": plan.model,
        "n_requests": len(requests),
        "status": "pending",
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    append(manifest_path, pending_row)
    batch = client.messages.batches.create(requests=requests)
    submitted_row = {**pending_row, "status": "submitted", "batch_id": batch.id}
    append(manifest_path, submitted_row)
    _log.info("batch_submitted", batch_id=batch.id, n=len(requests),
              phase=plan.phase, model=plan.model)
    return batch.id
