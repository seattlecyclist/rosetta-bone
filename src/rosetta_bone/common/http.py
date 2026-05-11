"""httpx-based HTTP client with on-disk response cache and basic retry."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._max_retries = max_retries

    def _path_for(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.bin"

    def get_bytes(self, url: str) -> bytes:
        cached = self._path_for(url)
        if cached.exists():
            return cached.read_bytes()
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(url, follow_redirects=True)
                resp.raise_for_status()
                cached.write_bytes(resp.content)
                return resp.content
            except httpx.HTTPStatusError as e:
                # 4xx is a client error; retrying won't change the result.
                if e.response.status_code < 500:
                    raise
                wait = 2**attempt
                _log.warning("http_retry", url=url, attempt=attempt + 1, error=str(e),
                             sleep_s=wait, status=e.response.status_code)
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(wait)
            except httpx.TransportError as e:
                wait = 2**attempt
                _log.warning("http_retry", url=url, attempt=attempt + 1, error=str(e),
                             sleep_s=wait)
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(wait)
        raise RuntimeError("unreachable")
