import pytest

from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.sft.prompt_builder import (
    MIN_CHUNK_CHARS,
    build_messages,
    build_system_block,
)


def _chunk(pillar: Pillar, text: str) -> Chunk:
    return Chunk(id=f"{pillar.value}-1", source="s", pillar=pillar, text=text, metadata={})


def _ok_chunks() -> dict[Pillar, Chunk]:
    return {
        Pillar.SCIENCE: _chunk(Pillar.SCIENCE,
            "Canine olfaction relies on the vomeronasal organ to detect "
            "volatile organic compounds and pheromonal signals with precision."),
        Pillar.STYLE: _chunk(Pillar.STYLE,
            "I trotted by my master's side, with my heart full of joy and my "
            "tail wagging gently behind me as we walked down the lane."),
        Pillar.BEHAVIOR: _chunk(Pillar.BEHAVIOR,
            "When the doorbell rings the dog typically rushes to the door, "
            "tail wagging, sniffing the threshold for familiar scents."),
    }


def test_system_block_contains_strict_context_language():
    block = build_system_block(_ok_chunks())
    assert "Do NOT invent" in block
    assert "strictly" in block.lower()
    assert "<persona>" in block and "</persona>" in block
    assert "<contract>" in block and "</contract>" in block
    for tag in ("science", "style", "behavior"):
        assert f"<{tag}>" in block
        assert f"</{tag}>" in block


def test_system_block_includes_chunk_text():
    chunks = _ok_chunks()
    block = build_system_block(chunks)
    for c in chunks.values():
        assert c.text in block


def test_min_chunk_chars_enforced():
    chunks = _ok_chunks()
    chunks[Pillar.SCIENCE] = _chunk(Pillar.SCIENCE, "tiny")
    with pytest.raises(ValueError, match="too short"):
        build_system_block(chunks)


def test_build_messages_user_block_has_stimulus_and_form_and_angle():
    chunks = _ok_chunks()
    msgs = build_messages(
        chunks,
        stimulus="vet visit",
        angle="dog anxious in waiting room",
        form="diary",
        variation=0,
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user = msgs[1]["content"]
    assert "vet visit" in user
    assert "diary" in user
    # The angle hint must reach Claude — that's the whole point of the
    # retrieval-variation fix.
    assert "dog anxious in waiting room" in user
    assert "Angle:" in user


def test_build_messages_different_angles_produce_different_user_blocks():
    chunks = _ok_chunks()
    a = build_messages(
        chunks, stimulus="vet visit", angle="dog anxious in waiting room",
        form="diary", variation=0,
    )[1]["content"]
    b = build_messages(
        chunks, stimulus="vet visit", angle="dog excited for vet treat",
        form="diary", variation=0,
    )[1]["content"]
    assert a != b


def test_min_chunk_chars_constant():
    assert MIN_CHUNK_CHARS >= 100
