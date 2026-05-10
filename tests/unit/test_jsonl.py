from pathlib import Path

from rosetta_bone.common.jsonl import append, iter_jsonl, write_all


def test_write_and_iter(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    write_all(p, [{"a": 1}, {"a": 2}])
    rows = list(iter_jsonl(p))
    assert rows == [{"a": 1}, {"a": 2}]


def test_append(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    append(p, {"a": 1})
    append(p, {"a": 2})
    assert list(iter_jsonl(p)) == [{"a": 1}, {"a": 2}]


def test_iter_missing_file_returns_empty(tmp_path: Path):
    assert list(iter_jsonl(tmp_path / "nope.jsonl")) == []
