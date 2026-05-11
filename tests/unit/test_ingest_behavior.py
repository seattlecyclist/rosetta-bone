from rosetta_bone.storyteller.ingest.behavior import (
    _pick_correct_option,
    extract_text_rows,
)


def test_pick_correct_option_returns_stripped_text():
    options = ["A. first answer text", "B. second answer text", "C. third"]
    assert _pick_correct_option(options, "B") == "second answer text"


def test_pick_correct_option_missing_letter_returns_none():
    assert _pick_correct_option(["A. foo", "B. bar"], "Z") is None


def test_pick_correct_option_empty_inputs():
    assert _pick_correct_option(None, "A") is None
    assert _pick_correct_option([], "A") is None
    assert _pick_correct_option(["A. foo"], None) is None


def test_extract_text_rows_combines_question_and_correct_option():
    fake_rows = [
        {
            "idx": 0,
            "scene_name": "video01",
            "question_category": "behavior_profiling",
            "question": "How does the dog react to the doorbell?",
            "ground_truth": "B",
            "options": [
                "A. The dog ignores it.",
                "B. The dog rushes to the door, tail wagging, sniffing the threshold.",
                "C. The dog hides.",
            ],
        },
        {
            "idx": 1,
            "scene_name": "video02",
            "question": "Q only — no options",
            "ground_truth": "A",
            "options": [],
        },
    ]
    out = extract_text_rows(fake_rows)
    assert len(out) == 1
    assert out[0]["source"] == "pawgaze/pawgaze:0"
    assert "How does the dog react" in out[0]["text"]
    assert "rushes to the door" in out[0]["text"]
    assert out[0]["metadata"]["scene_name"] == "video01"
    assert out[0]["metadata"]["question_category"] == "behavior_profiling"
    assert out[0]["metadata"]["ground_truth"] == "B"
