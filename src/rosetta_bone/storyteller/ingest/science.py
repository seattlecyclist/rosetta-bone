"""EuropePMC fetcher for the 'science' pillar (canine olfaction papers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from rosetta_bone.common.http import CachedClient
from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)

DEFAULT_QUERY = (
    '(canine olfaction OR vomeronasal OR "dog scent" OR "olfactory bulb dog") '
    "AND OPEN_ACCESS:Y"
)
_SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def pdf_url_for(pmcid: str) -> str:
    return f"https://europepmc.org/articles/{pmcid}?pdf=render"


def parse_search_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in payload.get("resultList", {}).get("result", []):
        if "pmcid" not in r:
            continue
        out.append({
            "pmcid": r["pmcid"],
            "title": r.get("title", ""),
            "pubYear": r.get("pubYear"),
        })
    return out


def search_papers(
    client: CachedClient,
    *,
    query: str = DEFAULT_QUERY,
    page_size: int = 25,
) -> list[dict[str, Any]]:
    qs = urlencode({"query": query, "format": "json", "pageSize": page_size})
    url = f"{_SEARCH_BASE}?{qs}"
    body = client.get_bytes(url)
    return parse_search_results(json.loads(body.decode()))


def fetch_papers(
    client: CachedClient,
    raw_dir: Path,
    *,
    limit: int = 25,
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = search_papers(client, page_size=limit)
    out: list[Path] = []
    for r in rows:
        pdf_path = raw_dir / f"{r['pmcid']}.pdf"
        meta_path = raw_dir / f"{r['pmcid']}.json"
        if pdf_path.exists() and meta_path.exists():
            out.append(pdf_path)
            continue
        try:
            content = client.get_bytes(pdf_url_for(r["pmcid"]))
        except Exception as e:
            _log.warning("europepmc_pdf_failed", pmcid=r["pmcid"], error=str(e))
            continue
        pdf_path.write_bytes(content)
        meta_path.write_text(json.dumps(r))
        out.append(pdf_path)
    return out
