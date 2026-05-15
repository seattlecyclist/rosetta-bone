"""Read-only summaries of what landed in data/raw/{pillar}/.

Used by `rb ingest-inspect` for human-in-the-loop review between
`ingest` and `chunk`. Pure inspection — never mutates raw_dir.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosetta_bone.common.jsonl import iter_jsonl

# Title-keyword buckets for the science pillar's smell/hearing balance check.
# Lowercase substring match, applied in order; first hit wins. The "other"
# bucket catches papers whose titles don't surface either modality.
_SCIENCE_BUCKETS: list[tuple[str, re.Pattern[str]]] = [
    ("smell", re.compile(r"olfact|scent|nasal|vomeronasal|odou?r|sniff", re.I)),
    ("hearing", re.compile(
        r"hear|audit|cochlea|baer|deaf|noise|sound|acoust|pinna|ultrasonic|presbycus",
        re.I,
    )),
]


@dataclass(frozen=True)
class SciencePaper:
    pmcid: str
    title: str
    pub_year: int | None
    pdf_bytes: int


@dataclass(frozen=True)
class ScienceSummary:
    papers: list[SciencePaper]
    bucket_counts: dict[str, int]  # smell / hearing / other
    total_bytes: int

    def to_dict(self, show: int) -> dict[str, Any]:
        years = [p.pub_year for p in self.papers if p.pub_year is not None]
        years.sort()
        year_stats = {
            "min": years[0] if years else None,
            "median": years[len(years) // 2] if years else None,
            "max": years[-1] if years else None,
            "count_per_year": dict(sorted(Counter(years).items())),
        }
        sorted_papers = sorted(
            self.papers, key=lambda p: (p.pub_year or 0), reverse=True,
        )
        return {
            "pillar": "science",
            "total_papers": len(self.papers),
            "total_bytes": self.total_bytes,
            "year_stats": year_stats,
            "topic_buckets": self.bucket_counts,
            "papers": [
                {
                    "pmcid": p.pmcid, "title": p.title,
                    "pub_year": p.pub_year, "pdf_bytes": p.pdf_bytes,
                }
                for p in sorted_papers[:show]
            ],
        }


def _bucket(title: str) -> str:
    for name, pat in _SCIENCE_BUCKETS:
        if pat.search(title):
            return name
    return "other"


def summarize_science(raw_dir: Path) -> ScienceSummary:
    """Scan data/raw/science for {pmcid}.pdf + {pmcid}.json sidecars."""
    papers: list[SciencePaper] = []
    if raw_dir.exists():
        for meta_path in sorted(raw_dir.glob("*.json")):
            pdf_path = meta_path.with_suffix(".pdf")
            if not pdf_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            year = meta.get("pubYear")
            try:
                year_int = int(year) if year is not None else None
            except (TypeError, ValueError):
                year_int = None
            papers.append(SciencePaper(
                pmcid=meta.get("pmcid", meta_path.stem),
                title=meta.get("title", ""),
                pub_year=year_int,
                pdf_bytes=pdf_path.stat().st_size,
            ))
    bucket_counts: dict[str, int] = {"smell": 0, "hearing": 0, "other": 0}
    for p in papers:
        bucket_counts[_bucket(p.title)] += 1
    return ScienceSummary(
        papers=papers,
        bucket_counts=bucket_counts,
        total_bytes=sum(p.pdf_bytes for p in papers),
    )


@dataclass(frozen=True)
class StyleBook:
    book_id: str
    bytes: int
    line_count: int


@dataclass(frozen=True)
class StyleSummary:
    books: list[StyleBook]

    def to_dict(self, show: int) -> dict[str, Any]:
        return {
            "pillar": "style",
            "total_books": len(self.books),
            "total_bytes": sum(b.bytes for b in self.books),
            "books": [
                {"book_id": b.book_id, "bytes": b.bytes, "lines": b.line_count}
                for b in self.books[:show]
            ],
        }


def summarize_style(raw_dir: Path) -> StyleSummary:
    books: list[StyleBook] = []
    if raw_dir.exists():
        for txt_path in sorted(raw_dir.glob("*.txt")):
            text = txt_path.read_text(errors="replace")
            books.append(StyleBook(
                book_id=txt_path.stem,
                bytes=txt_path.stat().st_size,
                line_count=text.count("\n") + 1,
            ))
    return StyleSummary(books=books)


@dataclass(frozen=True)
class BehaviorSummary:
    total_rows: int
    category_counts: dict[str, int]
    sample_rows: list[dict[str, Any]]

    def to_dict(self, show: int) -> dict[str, Any]:
        return {
            "pillar": "behavior",
            "total_rows": self.total_rows,
            "category_counts": self.category_counts,
            "sample_rows": self.sample_rows[:show],
        }


def summarize_behavior(raw_dir: Path) -> BehaviorSummary:
    jsonl = raw_dir / "pawgaze.jsonl"
    if not jsonl.exists():
        return BehaviorSummary(total_rows=0, category_counts={}, sample_rows=[])
    rows: Iterable[dict[str, Any]] = iter_jsonl(jsonl)
    category_counter: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    total = 0
    for r in rows:
        total += 1
        meta = r.get("metadata") or {}
        category = meta.get("question_category") or meta.get("scene_name") or "uncategorized"
        category_counter[str(category)] += 1
        if len(samples) < 50:
            samples.append({
                "source": r.get("source"),
                "category": category,
                "text_preview": (r.get("text", "")[:200]).replace("\n", " "),
            })
    return BehaviorSummary(
        total_rows=total,
        category_counts=dict(category_counter.most_common()),
        sample_rows=samples,
    )


def format_science(s: ScienceSummary, show: int) -> str:
    if not s.papers:
        return "science: no papers found in raw_dir."
    years = [p.pub_year for p in s.papers if p.pub_year is not None]
    years.sort()
    yr_min = years[0] if years else "?"
    yr_med = years[len(years) // 2] if years else "?"
    yr_max = years[-1] if years else "?"
    mb = s.total_bytes / (1024 * 1024)
    lines = [
        f"science: {len(s.papers)} papers, {mb:.1f} MB on disk",
        f"  year range: {yr_min} .. {yr_med} (median) .. {yr_max}",
        f"  topic buckets (title keyword match):",
        f"    smell:   {s.bucket_counts['smell']:>4}",
        f"    hearing: {s.bucket_counts['hearing']:>4}",
        f"    other:   {s.bucket_counts['other']:>4}",
        "",
        f"  most recent {min(show, len(s.papers))} papers:",
    ]
    sorted_papers = sorted(
        s.papers, key=lambda p: (p.pub_year or 0), reverse=True,
    )
    for p in sorted_papers[:show]:
        kb = p.pdf_bytes // 1024
        title = p.title if len(p.title) <= 90 else p.title[:87] + "..."
        lines.append(f"    {p.pmcid:<11} {p.pub_year or '????'}  {kb:>5} KB  {title}")
    return "\n".join(lines)


def format_style(s: StyleSummary, show: int) -> str:
    if not s.books:
        return "style: no books found in raw_dir."
    total_bytes = sum(b.bytes for b in s.books)
    lines = [
        f"style: {len(s.books)} books, {total_bytes / 1024:.1f} KB on disk",
        "",
        f"  books:",
    ]
    for b in s.books[:show]:
        kb = b.bytes // 1024
        lines.append(f"    {b.book_id:<8}  {kb:>5} KB  {b.line_count:>6} lines")
    return "\n".join(lines)


def format_behavior(s: BehaviorSummary, show: int) -> str:
    if s.total_rows == 0:
        return "behavior: no rows found (expected pawgaze.jsonl)."
    lines = [
        f"behavior: {s.total_rows} rows from pawgaze.jsonl",
        "",
        f"  top categories:",
    ]
    for cat, n in list(s.category_counts.items())[:show]:
        lines.append(f"    {n:>5}  {cat}")
    return "\n".join(lines)
