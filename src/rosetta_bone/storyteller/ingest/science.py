"""EuropePMC fetcher for the 'science' pillar (canine olfaction + audition)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from rosetta_bone.common.http import CachedClient
from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)

# Two queries, fetched separately and pooled into raw/science/. A single
# union query ranks by total terms matched, so the audition clause's
# longer keyword list out-scores the olfaction clause and pushes smell
# papers off the first page. Two independent caps guarantee both
# modalities reach the corpus regardless of EuropePMC's relevance sort.
OLFACTION_QUERY = (
    '(canine olfaction OR vomeronasal OR "dog scent" OR "olfactory bulb dog") '
    "AND OPEN_ACCESS:Y"
)
AUDITION_QUERY = (
    "("
    '(dog OR dogs OR canine OR canines OR "Canis familiaris")'
    " AND ("
    '"auditory brainstem response" OR BAER OR audiogram OR audiometry'
    " OR cochlea OR cochlear OR pinna"
    ' OR "sound localization" OR "sound localisation"'
    ' OR "hearing range" OR "hearing threshold" OR "hearing loss"'
    ' OR presbycusis OR deafness OR "noise phobia" OR "noise sensitivity"'
    ' OR ultrasonic OR "high frequency hearing"'
    ")"
    ") AND OPEN_ACCESS:Y"
)
# Back-compat alias. Existing callers that pass `query=DEFAULT_QUERY`
# still work; new code should prefer the explicit pair above.
DEFAULT_QUERY = OLFACTION_QUERY

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
    page_size: int = 50,
) -> list[dict[str, Any]]:
    qs = urlencode({"query": query, "format": "json", "pageSize": page_size})
    url = f"{_SEARCH_BASE}?{qs}"
    body = client.get_bytes(url)
    return parse_search_results(json.loads(body.decode()))


def fetch_papers(
    client: CachedClient,
    raw_dir: Path,
    *,
    limit: int = 50,
    query: str | None = None,
) -> list[Path]:
    """Fetch open-access dog science papers into raw_dir.

    If `query` is given, runs that single query with the given limit.
    Otherwise pools from OLFACTION_QUERY and AUDITION_QUERY at half
    the limit each (rounded up for olfaction) so both sensory
    modalities reach the corpus regardless of relevance ranking.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    if query is not None:
        queries = [(query, limit)]
    else:
        olf_n = (limit + 1) // 2
        aud_n = limit // 2
        queries = [(OLFACTION_QUERY, olf_n), (AUDITION_QUERY, aud_n)]

    rows: list[dict[str, Any]] = []
    seen_pmcids: set[str] = set()
    for q, n in queries:
        for r in search_papers(client, query=q, page_size=n):
            if r["pmcid"] in seen_pmcids:
                continue
            seen_pmcids.add(r["pmcid"])
            rows.append(r)
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
