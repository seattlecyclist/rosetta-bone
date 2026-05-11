import httpx

from rosetta_bone.common.http import CachedClient


def test_cache_hit_avoids_second_request(tmp_path, monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"hello")

    transport = httpx.MockTransport(handler)
    client = CachedClient(cache_dir=tmp_path, transport=transport)

    a = client.get_bytes("https://example.test/x")
    b = client.get_bytes("https://example.test/x")

    assert a == b == b"hello"
    assert calls["n"] == 1


def test_cache_miss_fetches(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=request.url.path.encode())

    client = CachedClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    assert client.get_bytes("https://example.test/a") == b"/a"
    assert client.get_bytes("https://example.test/b") == b"/b"
