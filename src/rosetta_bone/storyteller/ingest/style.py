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
    271,   # Black Beauty — Anna Sewell (anthropomorphic narrator)
    27805, # The Wind in the Willows — Kenneth Grahame (animal narrators)
]

_START_RE = re.compile(r"^\*\*\* START OF .* \*\*\*\s*$", re.MULTILINE)
_END_RE = re.compile(r"^\*\*\* END OF .* \*\*\*\s*$", re.MULTILINE)


def strip_gutenberg(text: str) -> str:
    s = _START_RE.search(text)
    e = _END_RE.search(text)
    if s and e:
        return text[s.end() : e.start()].strip()
    return text.strip()


def _candidate_urls(book_id: int) -> list[str]:
    """Common Project Gutenberg text-file URL patterns.

    Books vary: some have only `-0.txt` (UTF-8), some only `.txt` (ASCII),
    some only the canonical `cache/epub/pg{id}.txt`. Try each in order.
    """
    return [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]


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
        body: str | None = None
        for url in _candidate_urls(book_id):
            try:
                _log.info("gutenberg_fetch", id=book_id, url=url)
                body = client.get_bytes(url).decode("utf-8", errors="replace")
                break
            except Exception as e:
                _log.debug("gutenberg_url_failed", id=book_id, url=url, error=str(e))
        if body is None:
            _log.warning(
                "gutenberg_skip_unavailable",
                id=book_id,
                hint="no working URL; remove from GUTENBERG_BOOK_IDS or verify the ID",
            )
            continue
        target.write_text(strip_gutenberg(body))
        out.append(target)
    return out
