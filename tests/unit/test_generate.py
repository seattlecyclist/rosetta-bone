from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.sft.generate import (
    BatchPlan,
    enforce_request_cap,
    plan_batch,
    submit_batch,
)


def _ok_chunks() -> dict[Pillar, Chunk]:
    return {
        p: Chunk(id=f"{p.value}-1", source="s", pillar=p,
                 text=("x" * 200), metadata={}) for p in Pillar
    }


def test_enforce_request_cap_under_ok():
    enforce_request_cap(count=500, cap=1000)


def test_enforce_request_cap_over_raises():
    with pytest.raises(ValueError, match="cap"):
        enforce_request_cap(count=2000, cap=1000)


def test_plan_batch_builds_one_request_per_pair():
    triples = [("vet visit", 0, "diary"), ("vet visit", 1, "diary"),
               ("mailman", 0, "vignette")]
    select_calls = []

    def fake_select(stim):
        select_calls.append(stim)
        return _ok_chunks()

    plan = plan_batch(triples, select_fn=fake_select, model="claude-sonnet-4-6", phase="pilot")
    assert isinstance(plan, BatchPlan)
    assert len(plan.requests) == 3
    assert plan.requests[0].custom_id == "pilot::vet-visit::0"
    assert plan.requests[1].custom_id == "pilot::vet-visit::1"
    assert plan.requests[2].custom_id == "pilot::mailman::0"
    # Per-stimulus retrieval cache: only 2 distinct stimuli, so 2 select calls
    assert len(select_calls) == 2


def test_submit_batch_writes_manifest_before_call(tmp_path: Path):
    triples = [("vet visit", 0, "diary")]
    plan = plan_batch(triples, select_fn=lambda s: _ok_chunks(),
                      model="claude-sonnet-4-6", phase="pilot")
    fake_client = MagicMock()
    fake_client.messages.batches.create.return_value = MagicMock(id="msgbatch_xyz")

    manifest = tmp_path / "manifest.jsonl"
    bid = submit_batch(plan, client=fake_client, manifest_path=manifest)

    assert bid == "msgbatch_xyz"
    rows = list(iter_jsonl(manifest))
    # At least one row before submit (pending), one after (submitted)
    assert any(r["status"] == "submitted" and r["batch_id"] == "msgbatch_xyz" for r in rows)
    assert any(r["status"] == "pending" for r in rows)
    submitted = next(r for r in rows if r["status"] == "submitted")
    assert submitted["phase"] == "pilot"
    assert submitted["n_requests"] == 1
