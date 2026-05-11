import json
from pathlib import Path

from rosetta_bone.common.jsonl import iter_jsonl, write_all
from rosetta_bone.storyteller.sft.merge import (
    grounding_5gram_ratio,
    merge,
    parse_assistant_json,
)


def test_parse_valid_json():
    pair = parse_assistant_json('{"instruction":"i","story":"s"}')
    assert pair == {"instruction": "i", "story": "s"}


def test_parse_with_surrounding_text():
    raw = 'Here is the JSON: {"instruction":"i","story":"s"} thanks!'
    pair = parse_assistant_json(raw)
    assert pair == {"instruction": "i", "story": "s"}


def test_parse_invalid_returns_none():
    assert parse_assistant_json("not json") is None
    assert parse_assistant_json('{"missing_keys":1}') is None


def test_merge_dedupes_and_splits(tmp_path: Path):
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    write_all(batches_dir / "b1.jsonl", [
        {"custom_id": "p::a::0", "type": "succeeded",
         "text": json.dumps({"instruction": "I1", "story": "S1"})},
        {"custom_id": "p::a::1", "type": "succeeded",
         "text": json.dumps({"instruction": "I1", "story": "S1b"})},
        {"custom_id": "p::b::0", "type": "succeeded",
         "text": json.dumps({"instruction": "I2", "story": "S2"})},
        {"custom_id": "p::c::0", "type": "errored", "error": "boom"},
    ])

    train_p = tmp_path / "train.jsonl"
    valid_p = tmp_path / "valid.jsonl"
    stats = merge(batches_dir=batches_dir, train_path=train_p, valid_path=valid_p,
                  valid_fraction=0.5, seed=42)
    rows = list(iter_jsonl(train_p)) + list(iter_jsonl(valid_p))
    assert len(rows) == 2
    assert stats.kept == 2
    assert stats.dropped_invalid >= 1
    assert stats.deduped == 1
    assert all("messages" in r for r in rows)


def test_grounding_5gram_ratio_high_when_overlap():
    science_text = "the vomeronasal organ in dogs detects volatile compounds"
    story = "the dog used the vomeronasal organ in dogs detects volatile compounds today"
    assert grounding_5gram_ratio(story, science_text) > 0


def test_grounding_5gram_ratio_zero_when_no_overlap():
    assert grounding_5gram_ratio("totally unrelated text",
                                 "different content entirely") == 0.0
