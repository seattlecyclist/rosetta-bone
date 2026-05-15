"""S3-compatible storage client for dataset/adapter round-trip.

Targets Cloudflare R2 in production but works against any S3-API
endpoint (real S3, MinIO, moto's mock). The surface is intentionally
small — directory upload/download, prefix existence, list — because
that's all the orchestrator needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class S3Storage:
    """Thin boto3 wrapper bound to one bucket on an S3-compatible endpoint."""

    bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    # R2 ignores region but boto3 requires one. "auto" is what Cloudflare
    # tells you to use; real AWS S3 would need the real region.
    region_name: str = "auto"

    @classmethod
    def from_env(cls, bucket: str, endpoint_url: str) -> S3Storage:
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not access_key_id or not secret_access_key:
            raise RuntimeError(
                "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set "
                "in the environment to use remote training.",
            )
        return cls(
            bucket=bucket,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

    def _client(self):  # type: ignore[no-untyped-def]
        # Build per-call rather than caching on the instance: the dataclass
        # is frozen, and boto3 clients are cheap to construct (the
        # expensive part is the underlying urllib3 pool, which boto3
        # itself caches behind the scenes).
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

    def upload_dir(self, local: Path, prefix: str) -> int:
        """Upload every file under `local` to `<prefix>/<relpath>`.

        Returns the number of files uploaded. Symlinks are followed.
        Directories with no files produce no objects (S3 has no
        empty-folder concept).
        """
        prefix = prefix.rstrip("/")
        client = self._client()
        count = 0
        for path in _iter_files(local):
            relpath = path.relative_to(local).as_posix()
            key = f"{prefix}/{relpath}" if prefix else relpath
            client.upload_file(str(path), self.bucket, key)
            count += 1
        return count

    def download_dir(self, prefix: str, local: Path) -> int:
        """Mirror every object under `prefix` into `local/<relpath>`.

        Returns the number of objects downloaded.
        """
        prefix = prefix.rstrip("/") + "/"
        client = self._client()
        local.mkdir(parents=True, exist_ok=True)
        count = 0
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                relpath = key[len(prefix):]
                if not relpath:
                    continue
                dest = local / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(self.bucket, key, str(dest))
                count += 1
        return count

    def exists(self, prefix: str) -> bool:
        """True if any object exists under `prefix`."""
        prefix = prefix.rstrip("/") + "/"
        resp = self._client().list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=1,
        )
        return bool(resp.get("Contents"))

    def list_prefix(self, prefix: str) -> list[str]:
        """Return all object keys under `prefix`, sorted."""
        prefix = prefix.rstrip("/") + "/"
        keys: list[str] = []
        paginator = self._client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                keys.append(obj["Key"])
        return sorted(keys)


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path
