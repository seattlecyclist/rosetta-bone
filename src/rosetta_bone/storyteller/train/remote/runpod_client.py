"""Thin wrapper around the `runpod` SDK.

Surfaces just enough of the pod lifecycle for one-shot training jobs:
launch, poll until exit, fetch logs, terminate. We deliberately avoid
the SDK's higher-level abstractions (Endpoint, AsyncioJob) — they're
built for serverless inference, not for "run this container to
completion and tear it down."

The SDK's status strings have shifted over releases; rather than pin
to a specific enum, the wait loop treats anything that isn't an
active-running status as a terminal state and returns whatever exit
metadata is present.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

import runpod

# Statuses that mean "pod is still trying to do work." Anything else
# (EXITED, TERMINATED, FAILED, COMPLETED, …) we treat as terminal.
_ACTIVE_STATUSES = frozenset({"CREATED", "STARTING", "RUNNING", "RESTARTING"})


class RunPodError(RuntimeError):
    """Raised for non-transient RunPod API failures."""


@dataclass(frozen=True)
class PodHandle:
    pod_id: str
    name: str


@dataclass(frozen=True)
class ExitStatus:
    pod_id: str
    status: str
    exit_code: int | None
    runtime_seconds: float


class RunPodClient:
    """Pod lifecycle for one-shot containerised jobs."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise RunPodError(
                "RUNPOD_API_KEY must be set in the environment (or passed "
                "explicitly) to use remote training.",
            )
        runpod.api_key = key

    def launch_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_type: str,
        env: dict[str, str],
        container_disk_gb: int = 40,
        cloud_type: str = "COMMUNITY",
    ) -> PodHandle:
        """Create a pod that runs the image's ENTRYPOINT to completion.

        `container_disk_gb` needs to be big enough to hold the base
        model download (Llama-3.1-8B in bf16 ≈ 16 GB) plus working
        space; 40 GB is comfortable for the storyteller LoRA. Bump
        if you switch to a larger base.
        """
        result = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type,
            cloud_type=cloud_type,
            container_disk_in_gb=container_disk_gb,
            env=env,
            support_public_ip=False,
            start_ssh=False,
        )
        pod_id = result.get("id")
        if not pod_id:
            raise RunPodError(f"create_pod returned no id: {result!r}")
        return PodHandle(pod_id=pod_id, name=name)

    def wait_for_completion(
        self,
        handle: PodHandle,
        *,
        timeout_seconds: int,
        poll_interval_seconds: float = 10.0,
    ) -> ExitStatus:
        """Poll until the pod reaches a terminal status or we time out.

        On timeout, terminates the pod and raises. The caller is
        expected to inspect ExitStatus.exit_code; non-zero means the
        training job failed (logs are the diagnostic).
        """
        started = time.monotonic()
        while True:
            info = runpod.get_pod(handle.pod_id) or {}
            status = (info.get("desiredStatus") or info.get("status") or "").upper()
            if status and status not in _ACTIVE_STATUSES:
                return ExitStatus(
                    pod_id=handle.pod_id,
                    status=status,
                    exit_code=info.get("exitCode"),
                    runtime_seconds=time.monotonic() - started,
                )
            if time.monotonic() - started > timeout_seconds:
                self.terminate(handle)
                raise RunPodError(
                    f"pod {handle.pod_id} did not finish within "
                    f"{timeout_seconds}s (last status: {status or 'UNKNOWN'})",
                )
            time.sleep(poll_interval_seconds)

    def terminate(self, handle: PodHandle) -> None:
        """Best-effort terminate. Already-terminated pods are not an error."""
        # Already gone, or transient API blip. Either way we don't want
        # to mask a real upstream error with a teardown error.
        with contextlib.suppress(Exception):
            runpod.terminate_pod(handle.pod_id)
