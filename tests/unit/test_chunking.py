from rosetta_bone.common.chunking import chunk_text, count_tokens
from rosetta_bone.common.types import Pillar


def _gen_paragraphs(n: int, words_each: int = 200) -> str:
    return "\n\n".join(" ".join(f"word{j}" for j in range(words_each)) for _ in range(n))


def test_count_tokens_nonzero():
    assert count_tokens("hello world") > 0


def test_short_text_one_chunk():
    chunks = list(chunk_text("hello world", source_id="src1", pillar=Pillar.SCIENCE,
                             metadata={}, target_tokens=600, overlap=80))
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].pillar == Pillar.SCIENCE


def test_long_text_splits_with_overlap():
    text = _gen_paragraphs(20)  # well over 600 tokens
    chunks = list(chunk_text(text, source_id="src1", pillar=Pillar.STYLE,
                             metadata={"book": "x"}, target_tokens=300, overlap=50))
    assert len(chunks) >= 2
    assert chunks[0].id.startswith("src1-")
    assert chunks[1].id != chunks[0].id
    overlap_text = chunks[0].text[-100:]
    assert any(w in chunks[1].text for w in overlap_text.split()[:5])


def test_chunk_id_is_stable_hash():
    text = _gen_paragraphs(5)
    a = list(chunk_text(text, source_id="src1", pillar=Pillar.STYLE, metadata={}))
    b = list(chunk_text(text, source_id="src1", pillar=Pillar.STYLE, metadata={}))
    assert [c.id for c in a] == [c.id for c in b]
