"""Top-level orchestrator for `train --remote`.

Sequences the four stages:
    R1. upload dataset to R2 (idempotent — checks first)
    R2. launch RunPod pod with env-supplied job spec
    R3. pod runs train.py; we poll and stream logs back at the end
    R4. download PEFT adapter from R2; convert to mlx-lm format

If R2 already has the target adapter (content-addressed by data +
hyperparams + base model), we skip straight to R4. Re-running with
the same inputs is therefore free.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rosetta_bone.common.config import RemoteTrain
from rosetta_bone.storyteller.train.remote.convert import peft_to_mlx
from rosetta_bone.storyteller.train.remote.keys import (
    adapter_key,
    adapter_prefix,
    dataset_prefix,
)
from rosetta_bone.storyteller.train.remote.runpod_client import (
    PodHandle,
    RunPodClient,
)
from rosetta_bone.storyteller.train.remote.storage import S3Storage


class StorageProtocol(Protocol):
    """Subset of S3Storage that the orchestrator depends on (for mocking)."""

    def upload_dir(self, local: Path, prefix: str) -> int: ...
    def download_dir(self, prefix: str, local: Path) -> int: ...
    def exists(self, prefix: str) -> bool: ...


class RunPodProtocol(Protocol):
    """Subset of RunPodClient used by the orchestrator."""

    def launch_pod(
        self, *, name: str, image: str, gpu_type: str, env: dict[str, str],
    ) -> PodHandle: ...
    def wait_for_completion(
        self, handle: PodHandle, *, timeout_seconds: int,
    ) -> Any: ...
    def terminate(self, handle: PodHandle) -> None: ...


@dataclass(frozen=True)
class RemoteResult:
    """What `remote_train()` returns. Plumbed into the CLI's metadata.json."""

    adapter_dir: Path  # local dir containing adapters.safetensors + adapter_config.json
    adapter_key: str
    train_sha1: str
    valid_sha1: str
    pod_id: str
    gpu_type: str
    image: str
    pod_seconds: float
    short_circuit: bool  # true if we skipped training because the adapter already existed


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(65536), b""):
            h.update(buf)
    return h.hexdigest()


def remote_train(
    *,
    remote_cfg: RemoteTrain,
    train_path: Path,
    valid_path: Path,
    adapter_dir: Path,
    hyperparams: dict[str, Any],
    storage: StorageProtocol | None = None,
    runpod_client: RunPodProtocol | None = None,
) -> RemoteResult:
    """Run one LoRA fine-tune on RunPod and land the converted adapter locally.

    `storage` and `runpod_client` are injected for tests. In production
    both default to instances configured from `remote_cfg` + env.
    """
    storage = storage or S3Storage.from_env(
        bucket=remote_cfg.bucket, endpoint_url=remote_cfg.endpoint_url,
    )
    runpod_client = runpod_client or RunPodClient()

    train_sha = _sha1(train_path)
    valid_sha = _sha1(valid_path)
    key = adapter_key(
        train_sha1=train_sha, valid_sha1=valid_sha,
        base_model=remote_cfg.base_model, hyperparams=hyperparams,
    )
    ds_prefix = dataset_prefix(train_sha, valid_sha)
    ad_prefix = adapter_prefix(key) + "/peft"

    short_circuit = storage.exists(ad_prefix)
    pod_id = ""
    pod_seconds = 0.0

    if not short_circuit:
        if not storage.exists(ds_prefix):
            staging = Path(tempfile.mkdtemp(prefix="rb-remote-ds-"))
            try:
                (staging / "train.jsonl").write_bytes(train_path.read_bytes())
                (staging / "valid.jsonl").write_bytes(valid_path.read_bytes())
                storage.upload_dir(staging, ds_prefix)
            finally:
                _rmtree(staging)

        env = {
            "R2_ENDPOINT_URL": remote_cfg.endpoint_url,
            "R2_BUCKET": remote_cfg.bucket,
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "BASE_MODEL": remote_cfg.base_model,
            "DATASET_PREFIX": ds_prefix,
            "ADAPTER_PREFIX": ad_prefix,
            "HYPERPARAMS_JSON": json.dumps(hyperparams, sort_keys=True),
            "MAX_TRAIN_SECONDS": str(remote_cfg.pod_timeout_seconds - 60),
        }

        handle = runpod_client.launch_pod(
            name=f"rb-train-{key}",
            image=remote_cfg.image,
            gpu_type=remote_cfg.gpu_type,
            env=env,
        )
        started = time.monotonic()
        try:
            status = runpod_client.wait_for_completion(
                handle, timeout_seconds=remote_cfg.pod_timeout_seconds,
            )
        except Exception:
            runpod_client.terminate(handle)
            raise
        pod_seconds = time.monotonic() - started
        pod_id = handle.pod_id

        if status.exit_code not in (0, None):
            raise RuntimeError(
                f"remote training pod {pod_id} exited with code "
                f"{status.exit_code} (status={status.status}). "
                f"Check the pod's logs on RunPod.",
            )

    # R4: pull the PEFT adapter and convert it to mlx-lm format in-place
    # under the versioned adapter dir. We download to a tempdir first so
    # that a conversion failure doesn't leave a half-populated
    # versioned_dir for `latest` to point at.
    with tempfile.TemporaryDirectory(prefix="rb-remote-peft-") as tmp_str:
        tmp = Path(tmp_str)
        n = storage.download_dir(ad_prefix, tmp)
        if n == 0:
            raise RuntimeError(
                f"adapter prefix {ad_prefix} is empty after pod exit",
            )
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_to_mlx(tmp, adapter_dir)
        # Copy the in-pod train.log over so `train-inspect` works.
        pod_log = tmp / "train.log"
        if pod_log.exists():
            (adapter_dir / "train.log").write_bytes(pod_log.read_bytes())

    return RemoteResult(
        adapter_dir=adapter_dir,
        adapter_key=key,
        train_sha1=train_sha,
        valid_sha1=valid_sha,
        pod_id=pod_id,
        gpu_type=remote_cfg.gpu_type,
        image=remote_cfg.image,
        pod_seconds=pod_seconds,
        short_circuit=short_circuit,
    )


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
