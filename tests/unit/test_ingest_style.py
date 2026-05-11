from rosetta_bone.storyteller.ingest.style import GUTENBERG_BOOK_IDS, strip_gutenberg


def test_book_ids_are_pampered_pet():
    # Beautiful Joe (#440), A Dog's Tale (#1059), Bob Son of Battle (#3007)
    assert 440 in GUTENBERG_BOOK_IDS
    assert 1059 in GUTENBERG_BOOK_IDS
    assert 3007 in GUTENBERG_BOOK_IDS


def test_strip_header_footer():
    raw = (
        "preamble\n"
        "*** START OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
        "the actual book\n"
        "more content\n"
        "*** END OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
        "license blah\n"
    )
    body = strip_gutenberg(raw)
    assert "the actual book" in body
    assert "preamble" not in body
    assert "license blah" not in body
