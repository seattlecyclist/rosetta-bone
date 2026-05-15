from __future__ import annotations

from unittest.mock import patch

import pytest

from rosetta_bone.storyteller.train.remote.runpod_client import (
    PodHandle,
    RunPodClient,
    RunPodError,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunPodError, match="RUNPOD_API_KEY"):
        RunPodClient()


def test_launch_pod_passes_through_to_sdk():
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.create_pod",
        return_value={"id": "pod-xyz"},
    ) as create:
        client = RunPodClient()
        handle = client.launch_pod(
            name="rb-train-test",
            image="ghcr.io/rosetta-bone-trainer:v1",
            gpu_type="NVIDIA GeForce RTX 4090",
            env={"DATASET_PREFIX": "datasets/abc"},
        )
    assert handle.pod_id == "pod-xyz"
    assert handle.name == "rb-train-test"
    kwargs = create.call_args.kwargs
    assert kwargs["image_name"] == "ghcr.io/rosetta-bone-trainer:v1"
    assert kwargs["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert kwargs["env"] == {"DATASET_PREFIX": "datasets/abc"}
    assert kwargs["cloud_type"] == "COMMUNITY"


def test_launch_pod_raises_if_sdk_returns_no_id():
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.create_pod",
        return_value={},
    ), pytest.raises(RunPodError, match="no id"):
        RunPodClient().launch_pod(
            name="x", image="i", gpu_type="g", env={},
        )


def test_wait_for_completion_returns_on_exited():
    poll_responses = [
        {"desiredStatus": "RUNNING"},
        {"desiredStatus": "RUNNING"},
        {"desiredStatus": "EXITED", "exitCode": 0},
    ]
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.get_pod",
        side_effect=poll_responses,
    ), patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.time.sleep",
    ):
        client = RunPodClient()
        status = client.wait_for_completion(
            PodHandle(pod_id="p1", name="n"),
            timeout_seconds=60,
            poll_interval_seconds=0.0,
        )
    assert status.status == "EXITED"
    assert status.exit_code == 0


def test_wait_for_completion_terminates_and_raises_on_timeout():
    monotonic_values = iter([0.0, 5.0, 70.0, 70.0])
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.get_pod",
        return_value={"desiredStatus": "RUNNING"},
    ), patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.terminate_pod",
    ) as terminate, patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.time.monotonic",
        side_effect=lambda: next(monotonic_values),
    ), patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.time.sleep",
    ):
        client = RunPodClient()
        with pytest.raises(RunPodError, match="did not finish within"):
            client.wait_for_completion(
                PodHandle(pod_id="p1", name="n"),
                timeout_seconds=60,
                poll_interval_seconds=0.0,
            )
    terminate.assert_called_once_with("p1")


def test_wait_for_completion_propagates_non_zero_exit_code():
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.get_pod",
        return_value={"desiredStatus": "FAILED", "exitCode": 137},
    ), patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.time.sleep",
    ):
        client = RunPodClient()
        status = client.wait_for_completion(
            PodHandle(pod_id="p1", name="n"),
            timeout_seconds=60,
            poll_interval_seconds=0.0,
        )
    assert status.status == "FAILED"
    assert status.exit_code == 137


def test_terminate_swallows_sdk_errors():
    with patch(
        "rosetta_bone.storyteller.train.remote.runpod_client.runpod.terminate_pod",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise.
        RunPodClient().terminate(PodHandle(pod_id="p1", name="n"))
