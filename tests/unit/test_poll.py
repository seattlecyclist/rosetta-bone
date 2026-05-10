from pathlib import Path
from unittest.mock import MagicMock

from rosetta_bone.common.jsonl import append, iter_jsonl
from rosetta_bone.storyteller.sft.poll import PENDING_BATCH, poll_once


def test_poll_skips_completed_batches(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    append(manifest, {"phase": "pilot", "model": "x", "n_requests": 1,
                      "status": "submitted", "batch_id": "b1"})
    append(manifest, {"phase": "pilot", "model": "x", "n_requests": 1,
                      "status": "downloaded", "batch_id": "b1"})

    client = MagicMock()
    out_dir = tmp_path / "batches"
    pending = poll_once(client=client, manifest_path=manifest, out_dir=out_dir)
    assert pending == []
    client.messages.batches.retrieve.assert_not_called()


def test_poll_downloads_ended_batch(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    append(manifest, {"phase": "pilot", "model": "x", "n_requests": 2,
                      "status": "submitted", "batch_id": "b1"})

    client = MagicMock()
    client.messages.batches.retrieve.return_value = MagicMock(
        processing_status="ended"
    )
    client.messages.batches.results.return_value = iter([
        MagicMock(custom_id="pilot::vet::0", result=MagicMock(
            type="succeeded",
            message=MagicMock(
                content=[MagicMock(text='{"instruction": "i", "story": "s"}')],
                usage=MagicMock(input_tokens=1, output_tokens=2,
                                cache_read_input_tokens=0,
                                cache_creation_input_tokens=0),
            ),
        )),
    ])
    out_dir = tmp_path / "batches"
    pending = poll_once(client=client, manifest_path=manifest, out_dir=out_dir)
    assert pending == []
    assert (out_dir / "b1.jsonl").exists()
    rows = list(iter_jsonl(out_dir / "b1.jsonl"))
    assert rows[0]["custom_id"] == "pilot::vet::0"
    assert rows[0]["text"] == '{"instruction": "i", "story": "s"}'


def test_poll_keeps_in_progress(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    append(manifest, {"phase": "pilot", "model": "x", "n_requests": 2,
                      "status": "submitted", "batch_id": "b1"})

    client = MagicMock()
    client.messages.batches.retrieve.return_value = MagicMock(
        processing_status="in_progress"
    )
    pending = poll_once(client=client, manifest_path=manifest, out_dir=tmp_path / "out")
    assert pending == [PENDING_BATCH("b1", "in_progress")]
