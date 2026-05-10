from rosetta_bone.common.types import Chunk, Pillar


def test_pillar_values():
    assert Pillar.SCIENCE.value == "science"
    assert Pillar.STYLE.value == "style"
    assert Pillar.BEHAVIOR.value == "behavior"


def test_chunk_round_trip():
    c = Chunk(
        id="sci-pmc1-0",
        source="PMC1",
        pillar=Pillar.SCIENCE,
        text="hello",
        metadata={"year": 2020},
    )
    dumped = c.model_dump_json()
    c2 = Chunk.model_validate_json(dumped)
    assert c == c2
