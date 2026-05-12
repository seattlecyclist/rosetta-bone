import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.sft.generate import (
    _SLUG_MAX_LEN,
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


def test_slug_cap_is_30_for_angle_aware_custom_ids():
    # Four-component custom_id leaves less room for the slug; cap was
    # dropped from 40 -> 30 to keep us under Anthropic's 64-char limit.
    assert _SLUG_MAX_LEN == 30


def test_plan_batch_builds_one_request_per_pair_and_caches_per_angle():
    # (stimulus, embed_query, variation, form)
    pairs = [
        ("vet visit", "dog anxious at the vet", 0, "diary"),
        ("vet visit", "dog anxious at the vet", 1, "diary"),
        ("vet visit", "dog excited for vet treat", 0, "diary"),
        ("mailman", "dog barking at mailman", 0, "vignette"),
    ]
    select_calls = []

    def fake_select(embed_query):
        select_calls.append(embed_query)
        return _ok_chunks()

    plan = plan_batch(pairs, select_fn=fake_select,
                      model="claude-sonnet-4-6", phase="pilot")
    assert isinstance(plan, BatchPlan)
    assert len(plan.requests) == 4

    # Cache key is embed_query — 3 distinct queries -> 3 select calls.
    assert len(select_calls) == 3

    # custom_id = {phase}__{slug(stimulus)}__a{angle_idx}__v{var_idx}.
    # Angle index is per-stimulus, assigned first-seen.
    cids = [r.custom_id for r in plan.requests]
    assert cids[0] == "pilot__vet-visit__a0__v0"
    assert cids[1] == "pilot__vet-visit__a0__v1"
    assert cids[2] == "pilot__vet-visit__a1__v0"
    assert cids[3] == "pilot__mailman__a0__v0"

    # No two requests share a custom_id.
    assert len(set(cids)) == len(cids)

    # All match Anthropic's regex.
    for cid in cids:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid)


def test_plan_batch_angle_visible_in_user_block():
    pairs = [("x", "dog hiding in a closet", 0, "diary")]
    plan = plan_batch(
        pairs, select_fn=lambda q: _ok_chunks(),
        model="claude-sonnet-4-6", phase="p",
    )
    user_block = next(
        m["content"] for m in plan.requests[0].messages if m["role"] == "user"
    )
    assert "dog hiding in a closet" in user_block
    assert "Angle:" in user_block


def test_submit_batch_writes_manifest_before_call(tmp_path: Path):
    pairs = [("vet visit", "dog anxious at the vet", 0, "diary")]
    plan = plan_batch(pairs, select_fn=lambda q: _ok_chunks(),
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
