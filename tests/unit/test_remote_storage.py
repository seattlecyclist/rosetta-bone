"""Storage tests using moto's mocked S3.

moto is endpoint-agnostic: we point boto3 at moto's in-memory backend
via mock_aws() and verify the same surface we'll use against R2 in
production. The S3 API itself is what we're testing — credentials,
endpoint URL, and region are inert under the mock.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from rosetta_bone.storyteller.train.remote.storage import S3Storage

BUCKET = "rosetta-bone-test"


@pytest.fixture
def storage() -> S3Storage:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            endpoint_url="https://s3.amazonaws.com",
            access_key_id="test",
            secret_access_key="test",
            region_name="us-east-1",
        )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_upload_dir_uploads_every_file(tmp_path, storage):
    _write(tmp_path / "train.jsonl", '{"a": 1}\n')
    _write(tmp_path / "valid.jsonl", '{"b": 2}\n')
    _write(tmp_path / "sub" / "meta.json", "{}")

    n = storage.upload_dir(tmp_path, prefix="datasets/abc")

    assert n == 3
    assert set(storage.list_prefix("datasets/abc")) == {
        "datasets/abc/sub/meta.json",
        "datasets/abc/train.jsonl",
        "datasets/abc/valid.jsonl",
    }


def test_download_dir_is_inverse_of_upload(tmp_path, storage):
    src = tmp_path / "src"
    _write(src / "train.jsonl", '{"a": 1}\n')
    _write(src / "sub" / "meta.json", '{"x": 2}')
    storage.upload_dir(src, prefix="datasets/abc")

    dest = tmp_path / "dest"
    n = storage.download_dir("datasets/abc", dest)

    assert n == 2
    assert (dest / "train.jsonl").read_text() == '{"a": 1}\n'
    assert (dest / "sub" / "meta.json").read_text() == '{"x": 2}'


def test_exists_returns_true_only_when_objects_present(tmp_path, storage):
    assert storage.exists("adapters/missing") is False
    _write(tmp_path / "a.bin", "x")
    storage.upload_dir(tmp_path, prefix="adapters/present")
    assert storage.exists("adapters/present") is True
    assert storage.exists("adapters/missin") is False


def test_list_prefix_returns_sorted_keys(tmp_path, storage):
    _write(tmp_path / "z.txt", "z")
    _write(tmp_path / "a.txt", "a")
    _write(tmp_path / "m.txt", "m")
    storage.upload_dir(tmp_path, prefix="x")
    assert storage.list_prefix("x") == ["x/a.txt", "x/m.txt", "x/z.txt"]


def test_from_env_reads_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "SK")
    s = S3Storage.from_env(bucket="b", endpoint_url="https://example.com")
    assert s.access_key_id == "AK"
    assert s.secret_access_key == "SK"
    assert s.region_name == "auto"


def test_from_env_raises_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="R2_ACCESS_KEY_ID"):
        S3Storage.from_env(bucket="b", endpoint_url="https://example.com")
