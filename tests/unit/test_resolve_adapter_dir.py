"""Tests for the adapter-dir resolver in infer/model.py.

The resolver is what lets `generate` always read the most recent
training run via the `latest` symlink, while staying back-compatible
with the older single-directory layout.
"""

from pathlib import Path

from rosetta_bone.storyteller.infer.model import _resolve_adapter_dir


def test_returns_none_for_none_input():
    assert _resolve_adapter_dir(None) is None


def test_returns_none_for_empty_dir(tmp_path: Path):
    # Adapter root exists but has no `latest` and no *.safetensors.
    root = tmp_path / "adapter"
    root.mkdir()
    assert _resolve_adapter_dir(root) is None


def test_resolves_latest_symlink(tmp_path: Path):
    root = tmp_path / "adapter"
    versioned = root / "20260511T100000Z"
    versioned.mkdir(parents=True)
    (versioned / "adapters.safetensors").write_bytes(b"")
    (root / "latest").symlink_to("20260511T100000Z")

    resolved = _resolve_adapter_dir(root)
    assert resolved == root / "latest"
    # The symlink resolves to the timestamp directory.
    assert resolved.resolve() == versioned.resolve()


def test_legacy_layout_falls_back_to_root(tmp_path: Path):
    # Root directly contains adapter weights — no `latest` subdir.
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapters.safetensors").write_bytes(b"")

    assert _resolve_adapter_dir(root) == root


def test_latest_takes_precedence_over_root_weights(tmp_path: Path):
    # Pathological: weights at the root AND a `latest` symlink. The new
    # layout wins so the user is always reading from the versioned run.
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapters.safetensors").write_bytes(b"")
    versioned = root / "20260511T100000Z"
    versioned.mkdir()
    (root / "latest").symlink_to("20260511T100000Z")

    resolved = _resolve_adapter_dir(root)
    assert resolved == root / "latest"
