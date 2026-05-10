import json

import httpx

from rosetta_bone.common.http import CachedClient
from rosetta_bone.storyteller.ingest.science import (
    DEFAULT_QUERY,
    parse_search_results,
    pdf_url_for,
    search_papers,
)


def test_default_query_mentions_canine():
    assert "canine" in DEFAULT_QUERY.lower()
    assert "OPEN_ACCESS:Y" in DEFAULT_QUERY


def test_pdf_url_for_pmcid():
    assert pdf_url_for("PMC1234") == (
        "https://europepmc.org/articles/PMC1234?pdf=render"
    )


def test_parse_search_results():
    payload = {"resultList": {"result": [
        {"pmcid": "PMC123", "title": "T1", "pubYear": "2020"},
        {"pmcid": "PMC456", "title": "T2"},
        {"id": "no-pmcid"},
    ]}}
    rows = parse_search_results(payload)
    assert len(rows) == 2
    assert rows[0]["pmcid"] == "PMC123"
    assert rows[0]["title"] == "T1"
    assert rows[0]["pubYear"] == "2020"


def test_search_papers_uses_query(tmp_path):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"resultList": {"result": [
            {"pmcid": "PMC1", "title": "x"},
        ]}})

    client = CachedClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    rows = search_papers(client, query="foo", page_size=5)
    assert len(rows) == 1
    assert "query=foo" in captured["url"]
    assert "pageSize=5" in captured["url"]
    assert "format=json" in captured["url"]
