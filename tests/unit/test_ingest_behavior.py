from rosetta_bone.storyteller.ingest.behavior import extract_text_rows


def test_extract_text_rows_picks_text_columns():
    fake_rows = [
        {"id": 1, "description": "A small dog circles three times before lying down.",
         "label": "settling"},
        {"id": 2, "description": "", "label": "alert"},
        {"id": 3, "description": "Tail tucked, ears flattened.", "label": "fearful"},
    ]
    out = extract_text_rows(fake_rows, text_field="description")
    assert len(out) == 2
    assert "circles three times" in out[0]["text"]
    assert out[0]["source"] == "pawgaze/pawgaze:1"
    assert out[0]["metadata"]["label"] == "settling"
