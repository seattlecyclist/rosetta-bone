"""Project Gutenberg fetcher for the 'style' pillar (animal-POV fiction)."""

from __future__ import annotations

import re
from pathlib import Path

from rosetta_bone.common.http import CachedClient
from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)

# Curated for the lighthearted-pampered-pet persona of v1.
GUTENBERG_BOOK_IDS: list[int] = [
    440,   # Beautiful Joe — Marshall Saunders
    1059,  # A Dog's Tale — Mark Twain
    3007,  # Bob, Son of Battle — Alfred Ollivant
    23718, # Black Beauty (anthropomorphic narrator)
    19033, # Greyfriars Bobby
]

_START_RE = re.compile(r"^\*\*\* START OF .* \*\*\*\s*$", re.MULTILINE)
_END_RE = re.compile(r"^\*\*\* END OF .* \*\*\*\s*$", re.MULTILINE)


def strip_gutenberg(text: str) -> str:
    s = _START_RE.search(text)
    e = _END_RE.search(text)
    if s and e:
        return text[s.end() : e.start()].strip()
    return text.strip()


def fetch_books(client: CachedClient, raw_dir: Path, *, limit: int | None = None) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    ids = GUTENBERG_BOOK_IDS if limit is None else GUTENBERG_BOOK_IDS[:limit]
    for book_id in ids:
        target = raw_dir / f"{book_id}.txt"
        if target.exists():
            _log.debug("gutenberg_skip_existing", id=book_id)
            out.append(target)
            continue
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        _log.info("gutenberg_fetch", id=book_id, url=url)
        body = client.get_bytes(url).decode("utf-8", errors="replace")
        target.write_text(strip_gutenberg(body))
        out.append(target)
    return out
