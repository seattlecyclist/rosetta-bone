"""Tests for the pre-training stats inspector."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from rosetta_bone.common.jsonl import write_all
from rosetta_bone.storyteller.cli import app
from rosetta_bone.storyteller.sft.merge import _hash_instr
from rosetta_bone.storyteller.sft.stats import (
    _build_attribution_map,
    _count_persona_violations,
    _custom_id_parts,
    _quantiles,
    compute_stats,
    format_stats_table,
)
from rosetta_bone.storyteller.sft.stimuli import Stimulus

runner = CliRunner()


def test_custom_id_parts_roundtrip():
    parsed = _custom_id_parts("pilot__the-mailman-arriving__a0__v2")
    assert parsed == ("pilot", "the-mailman-arriving", 0, 2)


def test_custom_id_parts_handles_phase_with_separator():
    parsed = _custom_id_parts("pilot__v2__the-mailman-arriving__a1__v0")
    assert parsed == ("pilot__v2", "the-mailman-arriving", 1, 0)


def test_custom_id_parts_rejects_malformed():
    assert _custom_id_parts("malformed") is None
    assert _custom_id_parts("p__slug__xN__v0") is None
    assert _custom_id_parts("p__slug__a0__notavar") is None


def test_build_attribution_map():
    stims = [
        Stimulus(
            prompt="the mailman arriving",
            embed_queries=["aggressive at door", "hiding from mailman"],
            variations_per_query=2,
            form="diary",
        ),
        Stimulus(
            prompt="a trip to the vet",
            embed_queries=["anxious in waiting room"],
            variations_per_query=1,
            form="vignette",
        ),
    ]
    m = _build_attribution_map(stims)
    assert m[("the-mailman-arriving", 0)] == (
        "the mailman arriving", "aggressive at door", "diary",
    )
    assert m[("the-mailman-arriving", 1)] == (
        "the mailman arriving", "hiding from mailman", "diary",
    )
    assert m[("a-trip-to-the-vet", 0)] == (
        "a trip to the vet", "anxious in waiting room", "vignette",
    )


def test_count_persona_violations_finds_known_markers():
    assert _count_persona_violations("just normal prose") == 0
    assert _count_persona_violations(
        "a vast olfactory plume arrived from everywhere"
    ) >= 1
    assert _count_persona_violations("I contemplated the meaning") >= 1


def test_quantiles_handles_empty_and_one_element():
    assert _quantiles([]) == {"p10": 0, "p50": 0, "p90": 0, "max": 0}
    assert _quantiles([42]) == {"p10": 42, "p50": 42, "p90": 42, "max": 42}


def test_quantiles_basic():
    q = _quantiles(list(range(1, 101)))
    assert q["p50"] == 50
    assert q["p10"] <= 11 and q["p10"] >= 9
    assert q["p90"] >= 89
    assert q["max"] == 100


def _seed_corpus(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal batches dir + train/valid + stimuli yaml."""
    stim_path = tmp_path / "stimuli.yaml"
    stim_path.write_text(yaml.safe_dump([
        {"prompt": "the vet", "embed_queries": ["anxious", "treat"],
         "variations_per_query": 2, "form": "diary"},
    ]))

    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    # 4 generated, of which 3 will survive dedup (one is exact duplicate
    # of another, one is errored, one is invalid JSON).
    write_all(batches_dir / "b1.jsonl", [
        # Kept pair
        {"custom_id": "p__the-vet__a0__v0", "type": "succeeded",
         "text": json.dumps({"instruction": "I1 anxious", "story": "S1 anxious story"})},
        # Duplicate instruction (same hash) - merge will dedupe
        {"custom_id": "p__the-vet__a0__v1", "type": "succeeded",
         "text": json.dumps({"instruction": "I1 anxious", "story": "S1b"})},
        # Different angle, kept
        {"custom_id": "p__the-vet__a1__v0", "type": "succeeded",
         "text": json.dumps({"instruction": "I2 treat", "story": "S2 treat story"})},
        # Errored
        {"custom_id": "p__the-vet__a1__v1", "type": "errored", "error": "boom"},
        # Invalid JSON
        {"custom_id": "p__the-vet__a0__v0", "type": "succeeded",
         "text": "not even close to JSON"},
    ])

    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    write_all(train_path, [
        {"messages": [{"role": "user", "content": "I1 anxious"},
                      {"role": "assistant", "content": "S1 anxious story"}]},
    ])
    write_all(valid_path, [
        {"messages": [{"role": "user", "content": "I2 treat"},
                      {"role": "assistant", "content": "S2 treat story"}]},
    ])

    return {
        "stimuli": stim_path,
        "batches_dir": batches_dir,
        "train": train_path,
        "valid": valid_path,
    }


def test_compute_stats_summary_and_per_stimulus(tmp_path: Path):
    paths = _seed_corpus(tmp_path)
    stats = compute_stats(
        batches_dir=paths["batches_dir"],
        train_path=paths["train"],
        valid_path=paths["valid"],
        stimuli_path=paths["stimuli"],
    )

    s = stats["summary"]
    assert s["n_raw"] == 5
    assert s["n_errored"] == 1
    assert s["n_invalid_json"] == 1
    assert s["n_generated_valid"] == 3   # 5 - 1 errored - 1 invalid
    assert s["n_kept"] == 2              # I1 dup got deduped at merge

    ps = stats["per_stimulus"]
    assert len(ps) == 1
    vet = ps[0]
    assert vet["stimulus"] == "the vet"
    assert vet["n_generated"] == 3
    assert vet["n_kept"] == 2

    # Per-angle: a0 had 2 generated, 1 kept (dup dropped); a1 had 1, 1 kept.
    angles_by_text = {a["angle"]: a for a in vet["angles"]}
    assert angles_by_text["anxious"]["n_generated"] == 2
    assert angles_by_text["anxious"]["n_kept"] == 1
    assert angles_by_text["treat"]["n_generated"] == 1
    assert angles_by_text["treat"]["n_kept"] == 1


def test_format_stats_table_renders_known_sections(tmp_path: Path):
    paths = _seed_corpus(tmp_path)
    stats = compute_stats(
        batches_dir=paths["batches_dir"],
        train_path=paths["train"],
        valid_path=paths["valid"],
        stimuli_path=paths["stimuli"],
    )
    out = format_stats_table(stats)
    assert "SUMMARY" in out
    assert "PER-STIMULUS" in out
    assert "PER-ANGLE" in out
    assert "the vet" in out


def test_sft_stats_help_documents_options():
    r = runner.invoke(app, ["sft", "stats", "--help"])
    assert r.exit_code == 0
    assert "--output" in r.output
    assert "--stimuli" in r.output


def test_sft_help_lists_stats_subcommand():
    r = runner.invoke(app, ["sft", "--help"])
    assert r.exit_code == 0
    assert "stats" in r.output


def test_kept_hashes_uses_same_normalization_as_merge():
    """The stats joiner must use merge's _hash_instr so the
    kept-vs-not bookkeeping matches what actually got written
    to train/valid."""
    # The instruction text from the batch has trailing whitespace
    # that merge.py normalizes away. Stats must do the same.
    raw_instruction = "  I am an instruction  "
    normalized_hash = _hash_instr(raw_instruction)
    # If anyone changes _hash_instr's normalization, this test
    # will fail and stats will drift from merge.
    assert normalized_hash == _hash_instr("I am an instruction")
