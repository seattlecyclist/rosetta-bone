"""Orchestrator unit tests with fakes for storage + runpod.

We don't mock the converter — it's already covered by its own tests
and the integration of producing real PEFT bytes -> mlx-lm files is
exactly the seam we want exercised here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from rosetta_bone.common.config import RemoteTrain
from rosetta_bone.storyteller.train.remote.orchestrator import (
    PodHandle,
    remote_train,
)


@dataclass
class _ExitStatus:
    pod_id: str = "pod-1"
    status: str = "EXITED"
    exit_code: int | None = 0
    runtime_seconds: float = 1.0


class FakeStorage:
    """In-memory storage fake. Tracks calls so tests can assert sequencing."""

    def __init__(self) -> None:
        # prefix -> {relpath: bytes}
        self._objects: dict[str, dict[str, bytes]] = {}
        self.calls: list[str] = []

    def upload_dir(self, local: Path, prefix: str) -> int:
        self.calls.append(f"upload {prefix}")
        bucket = self._objects.setdefault(prefix.rstrip("/"), {})
        n = 0
        for path in sorted(local.rglob("*")):
            if path.is_file():
                rel = path.relative_to(local).as_posix()
                bucket[rel] = path.read_bytes()
                n += 1
        return n

    def download_dir(self, prefix: str, local: Path) -> int:
        self.calls.append(f"download {prefix}")
        local.mkdir(parents=True, exist_ok=True)
        bucket = self._objects.get(prefix.rstrip("/"), {})
        for rel, data in bucket.items():
            target = local / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return len(bucket)

    def exists(self, prefix: str) -> bool:
        return bool(self._objects.get(prefix.rstrip("/")))

    def seed_adapter(self, prefix: str, peft_dir: Path) -> None:
        bucket = self._objects.setdefault(prefix.rstrip("/"), {})
        for path in sorted(peft_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(peft_dir).as_posix()
                bucket[rel] = path.read_bytes()


class FakeRunPod:
    def __init__(self, exit_code: int = 0) -> None:
        self.launched: list[dict] = []
        self.terminated: list[str] = []
        self._exit_code = exit_code

    def launch_pod(self, *, name, image, gpu_type, env):  # type: ignore[no-untyped-def]
        self.launched.append({
            "name": name, "image": image, "gpu_type": gpu_type, "env": env,
        })
        return PodHandle(pod_id="pod-1", name=name)

    def wait_for_completion(self, handle, *, timeout_seconds):  # type: ignore[no-untyped-def]
        return _ExitStatus(pod_id=handle.pod_id, exit_code=self._exit_code,
                           status="EXITED" if self._exit_code == 0 else "FAILED")

    def terminate(self, handle: PodHandle) -> None:
        self.terminated.append(handle.pod_id)


def _make_peft_dir(tmp_path: Path) -> Path:
    """Build a tiny valid PEFT adapter we can pretend the pod produced."""
    class _MiniLlama(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.ModuleDict({
                "layers": nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": nn.ModuleDict({
                            "q_proj": nn.Linear(8, 8, bias=False),
                            "v_proj": nn.Linear(8, 8, bias=False),
                        }),
                    }),
                ]),
            })

        def forward(self, x):  # type: ignore[no-untyped-def]
            return x
    m = get_peft_model(_MiniLlama(), LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="none",
    ))
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.fill_(0.01)
    out = tmp_path / "fake-peft"
    m.save_pretrained(out)
    return out


@pytest.fixture
def remote_cfg() -> RemoteTrain:
    return RemoteTrain(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        image="ghcr.io/rosetta-bone-trainer:test",
        gpu_type="RTX 4090",
        bucket="rb-test",
        endpoint_url="https://example.com",
        pod_timeout_seconds=600,
    )


@pytest.fixture
def datasets(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text('{"messages": [{"role": "user", "content": "x"}]}\n')
    valid.write_text('{"messages": [{"role": "user", "content": "y"}]}\n')
    return train, valid


HP = {"rank": 4, "alpha": 8, "iters": 50}


def test_happy_path_uploads_data_launches_pod_downloads_adapter(
    monkeypatch, tmp_path, remote_cfg, datasets,
):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("HF_TOKEN", "ht")
    train, valid = datasets
    storage = FakeStorage()
    runpod = FakeRunPod(exit_code=0)

    # Pre-arrange: when the pod "runs", it would upload its adapter to
    # the expected prefix. Stage that here so download_dir finds bytes.
    # We need the right key first; compute it the same way the
    # orchestrator does, by triggering a dry exists() check.
    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_key as _ak,
    )
    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_prefix as _ap,
    )
    from rosetta_bone.storyteller.train.remote.orchestrator import _sha1
    key = _ak(
        train_sha1=_sha1(train), valid_sha1=_sha1(valid),
        base_model=remote_cfg.base_model, hyperparams=HP,
    )
    storage.seed_adapter(_ap(key) + "/peft", _make_peft_dir(tmp_path))
    # Force orchestrator to take the full path (not short-circuit): the
    # caller doesn't see the adapter as already-existing before pod
    # launch, so we keep `exists()` truthful via a tiny override:
    real_exists = storage.exists
    seen_once: dict[str, bool] = {}

    def first_call_false(prefix: str) -> bool:
        if prefix.startswith("adapters/") and not seen_once.get(prefix):
            seen_once[prefix] = True
            return False
        return real_exists(prefix)

    storage.exists = first_call_false  # type: ignore[method-assign]

    out_dir = tmp_path / "out"
    result = remote_train(
        remote_cfg=remote_cfg,
        train_path=train, valid_path=valid,
        adapter_dir=out_dir, hyperparams=HP,
        storage=storage, runpod_client=runpod,
    )

    assert result.short_circuit is False
    assert result.pod_id == "pod-1"
    assert (out_dir / "adapters.safetensors").exists()
    assert (out_dir / "adapter_config.json").exists()

    # Pod was launched with the right env contract.
    assert len(runpod.launched) == 1
    env = runpod.launched[0]["env"]
    assert env["BASE_MODEL"] == remote_cfg.base_model
    assert env["DATASET_PREFIX"].startswith("datasets/")
    assert env["ADAPTER_PREFIX"].startswith("adapters/")
    assert json.loads(env["HYPERPARAMS_JSON"])["rank"] == 4

    # Storage saw: exists(adapter), exists(dataset), upload(dataset),
    # exists(adapter again at success), download(adapter).
    assert any(c.startswith("upload datasets/") for c in storage.calls)
    assert any(c.startswith("download adapters/") for c in storage.calls)


def test_short_circuit_when_adapter_already_exists(
    monkeypatch, tmp_path, remote_cfg, datasets,
):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    train, valid = datasets
    storage = FakeStorage()
    runpod = FakeRunPod()

    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_key as _ak,
    )
    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_prefix as _ap,
    )
    from rosetta_bone.storyteller.train.remote.orchestrator import _sha1
    key = _ak(
        train_sha1=_sha1(train), valid_sha1=_sha1(valid),
        base_model=remote_cfg.base_model, hyperparams=HP,
    )
    storage.seed_adapter(_ap(key) + "/peft", _make_peft_dir(tmp_path))

    out_dir = tmp_path / "out"
    result = remote_train(
        remote_cfg=remote_cfg,
        train_path=train, valid_path=valid,
        adapter_dir=out_dir, hyperparams=HP,
        storage=storage, runpod_client=runpod,
    )

    assert result.short_circuit is True
    assert runpod.launched == []  # no pod was launched
    assert (out_dir / "adapters.safetensors").exists()


def test_non_zero_exit_raises_and_does_not_convert(
    monkeypatch, tmp_path, remote_cfg, datasets,
):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    train, valid = datasets
    storage = FakeStorage()
    runpod = FakeRunPod(exit_code=137)

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="exited with code 137"):
        remote_train(
            remote_cfg=remote_cfg,
            train_path=train, valid_path=valid,
            adapter_dir=out_dir, hyperparams=HP,
            storage=storage, runpod_client=runpod,
        )
    # No partial adapter dir left behind.
    assert not (out_dir / "adapters.safetensors").exists()


def test_terminates_pod_on_wait_exception(
    monkeypatch, tmp_path, remote_cfg, datasets,
):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    train, valid = datasets
    storage = FakeStorage()
    runpod = FakeRunPod()

    def boom(handle, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("wait crashed")
    runpod.wait_for_completion = boom  # type: ignore[method-assign]

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="wait crashed"):
        remote_train(
            remote_cfg=remote_cfg,
            train_path=train, valid_path=valid,
            adapter_dir=out_dir, hyperparams=HP,
            storage=storage, runpod_client=runpod,
        )
    assert runpod.terminated == ["pod-1"]


def test_reuses_dataset_when_already_uploaded(
    monkeypatch, tmp_path, remote_cfg, datasets,
):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    train, valid = datasets
    storage = FakeStorage()
    runpod = FakeRunPod()

    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_key as _ak,
    )
    from rosetta_bone.storyteller.train.remote.keys import (
        adapter_prefix as _ap,
    )
    from rosetta_bone.storyteller.train.remote.keys import (
        dataset_prefix as _dp,
    )
    from rosetta_bone.storyteller.train.remote.orchestrator import _sha1
    train_sha = _sha1(train)
    valid_sha = _sha1(valid)
    key = _ak(
        train_sha1=train_sha, valid_sha1=valid_sha,
        base_model=remote_cfg.base_model, hyperparams=HP,
    )
    # Pre-seed dataset (would-be cache hit) AND the eventual adapter.
    storage._objects[_dp(train_sha, valid_sha)] = {"train.jsonl": b"x"}
    storage.seed_adapter(_ap(key) + "/peft", _make_peft_dir(tmp_path))

    # Make the adapter look absent on the FIRST exists() check only.
    real_exists = storage.exists
    fired = {"adapter": False}

    def first_adapter_false(prefix: str) -> bool:
        if prefix.startswith("adapters/") and not fired["adapter"]:
            fired["adapter"] = True
            return False
        return real_exists(prefix)
    storage.exists = first_adapter_false  # type: ignore[method-assign]

    out_dir = tmp_path / "out"
    remote_train(
        remote_cfg=remote_cfg,
        train_path=train, valid_path=valid,
        adapter_dir=out_dir, hyperparams=HP,
        storage=storage, runpod_client=runpod,
    )

    # We did not upload the dataset (it was already there).
    assert not any(c.startswith("upload datasets/") for c in storage.calls)
