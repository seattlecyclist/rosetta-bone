"""Token-aware text splitter using tiktoken cl100k_base.

~600 tokens per chunk by default, ~80-token overlap, splitting on
paragraph (\\n\\n) then sentence (. ! ?) boundaries.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from typing import Any

import tiktoken

from rosetta_bone.common.types import Chunk, Pillar

_ENC = tiktoken.get_encoding("cl100k_base")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENT_RE.split(paragraph)
    return [s.strip() for s in parts if s.strip()]


def _accumulate(units: list[str], target_tokens: int) -> Iterator[str]:
    """Greedy pack `units` into chunks at most `target_tokens` tokens.

    Each output chunk is a join of one or more consecutive units. A unit
    that's individually larger than target_tokens is emitted on its own.
    """
    buf: list[str] = []
    buf_tok = 0
    for u in units:
        u_tok = count_tokens(u)
        if u_tok > target_tokens:
            if buf:
                yield " ".join(buf)
                buf, buf_tok = [], 0
            yield u
            continue
        if buf and buf_tok + u_tok > target_tokens:
            yield " ".join(buf)
            buf, buf_tok = [], 0
        buf.append(u)
        buf_tok += u_tok
    if buf:
        yield " ".join(buf)


def chunk_text(
    text: str,
    *,
    source_id: str,
    pillar: Pillar,
    metadata: dict[str, Any],
    target_tokens: int = 600,
    overlap: int = 80,
) -> Iterator[Chunk]:
    units: list[str] = []
    for para in _split_paragraphs(text):
        if count_tokens(para) <= target_tokens:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    if not units:
        return

    raw_chunks = list(_accumulate(units, target_tokens))

    # Apply overlap: prepend tail of previous chunk to current chunk.
    out: list[str] = []
    for i, c in enumerate(raw_chunks):
        if i == 0:
            out.append(c)
        else:
            prev_tail_tokens = _ENC.encode(out[-1])[-overlap:]
            tail = _ENC.decode(prev_tail_tokens)
            out.append(tail + " " + c)

    for i, t in enumerate(out):
        h = hashlib.sha1(f"{source_id}|{i}|{t[:64]}".encode()).hexdigest()[:10]
        yield Chunk(
            id=f"{source_id}-{i:04d}-{h}",
            source=source_id,
            pillar=pillar,
            text=t,
            metadata=metadata,
        )
