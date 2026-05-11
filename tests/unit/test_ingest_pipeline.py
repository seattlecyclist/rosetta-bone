from pathlib import Path

from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.types import Pillar
from rosetta_bone.storyteller.ingest.pipeline import chunk_pillar


def test_chunk_pillar_writes_jsonl(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "src1.txt").write_text("first paragraph.\n\nsecond paragraph.")
    (raw / "src2.txt").write_text("another short paragraph.")

    out = chunk_pillar(
        raw_dir=raw,
        pillar=Pillar.STYLE,
        out_path=tmp_path / "style.jsonl",
        target_tokens=600,
        overlap=80,
    )
    rows = list(iter_jsonl(out))
    assert len(rows) >= 2
    assert {r["source"] for r in rows} == {"src1", "src2"}
    assert all(r["pillar"] == "style" for r in rows)
