"""Content-addressed adapter keys.

The key is `sha256(canonical_json({data, model, hyperparams}))`,
hex-truncated. Same inputs → same key, so re-running `train --remote`
with an already-trained spec is free: the orchestrator finds the
adapter in R2 and skips straight to download.

GPU type is deliberately NOT part of the key — a 4090 and an A40
producing functionally equivalent adapters dedupe correctly. If we
ever want strict per-GPU reproducibility, add it here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

KEY_LENGTH = 16


def adapter_key(
    *,
    train_sha1: str,
    valid_sha1: str,
    base_model: str,
    hyperparams: dict[str, Any],
) -> str:
    """Return a 16-hex-char content-address for an adapter."""
    payload = {
        "train_sha1": train_sha1,
        "valid_sha1": valid_sha1,
        "base_model": base_model,
        "hyperparams": hyperparams,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:KEY_LENGTH]


def dataset_prefix(train_sha1: str, valid_sha1: str) -> str:
    """Prefix under which the dataset for this train/valid pair lives."""
    combined = hashlib.sha256(
        f"{train_sha1}:{valid_sha1}".encode(),
    ).hexdigest()[:KEY_LENGTH]
    return f"datasets/{combined}"


def adapter_prefix(key: str) -> str:
    return f"adapters/{key}"
