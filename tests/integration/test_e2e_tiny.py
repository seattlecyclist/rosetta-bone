"""End-to-end smoke test. Slow + costs ~$0.10 in API + downloads ~2GB.

Runs: ingest → chunk → embed → SFT generate (5 pairs) → poll → merge →
train (50 iters of Llama-3.2-3B-4bit) → infer once.

Run: uv run pytest tests/integration -m slow -v
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app

pytestmark = pytest.mark.slow

runner = CliRunner()


def _override_config_for_test(tmp_path: Path) -> Path:
    cfg = tmp_path / "test.toml"
    cfg.write_text(f"""
[paths]
data_dir = "{tmp_path}/data"
raw_dir = "{tmp_path}/data/raw"
chunks_dir = "{tmp_path}/data/chunks"
embeddings_dir = "{tmp_path}/data/embeddings"
sft_dir = "{tmp_path}/data/sft"
adapter_dir = "{tmp_path}/data/adapters/test"

[retrieval]
embedding_model = "BAAI/bge-small-en-v1.5"
similarity_threshold = 0.10

[sft]
model = "claude-sonnet-4-6"
max_requests_per_run = 10
requests_per_minute = 5
batch_size_max = 100

[train]
base_model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
rank = 4
alpha = 8.0
iters = 50
batch_size = 1
learning_rate = 1e-4
target_modules = ["q_proj", "v_proj"]

[infer]
temperature = 0.7
top_p = 0.9
max_tokens = 200
repetition_penalty = 1.05
""")
    return cfg


def _shrink_stimuli(tmp_path: Path) -> Path:
    p = tmp_path / "stimuli.yaml"
    p.write_text(yaml.safe_dump([
        {"prompt": "the mailman arriving", "variations": 2, "form": "diary"},
        {"prompt": "a trip to the vet", "variations": 2, "form": "vignette"},
        {"prompt": "dinner being prepared", "variations": 1, "form": "diary"},
    ]))
    return p


def test_pipeline_e2e(tmp_path: Path):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    cfg = _override_config_for_test(tmp_path)
    stimuli = _shrink_stimuli(tmp_path)

    conv = Path("config/stimuli.yaml")
    backup = None
    if conv.exists():
        backup = conv.with_suffix(".bak")
        conv.rename(backup)
    try:
        conv.write_text(stimuli.read_text())

        for pillar, limit in [("style", 1), ("science", 2), ("behavior", 20)]:
            r = runner.invoke(app, ["ingest", "--pillar", pillar,
                                    "--limit", str(limit), "--config", str(cfg)])
            assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["chunk", "--all", "--config", str(cfg)])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["embed", "--config", str(cfg)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["sft", "generate", "--count", "5",
                                "--phase", "smoke", "--config", str(cfg)])
        assert r.exit_code == 0, r.output

        deadline = time.time() + 30 * 60
        while time.time() < deadline:
            r = runner.invoke(app, ["sft", "poll", "--config", str(cfg)])
            assert r.exit_code == 0, r.output
            if "All batches downloaded" in r.output:
                break
            time.sleep(60)
        else:
            pytest.fail("Batch did not complete within 30 minutes")

        r = runner.invoke(app, ["sft", "merge", "--config", str(cfg)])
        assert r.exit_code == 0, r.output
        train_path = tmp_path / "data" / "sft" / "train.jsonl"
        assert train_path.exists() and train_path.stat().st_size > 0

        r = runner.invoke(app, ["train", "--iters", "50", "--config", str(cfg)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["generate", "a trip to the vet",
                                "--max-tokens", "100", "--config", str(cfg)])
        assert r.exit_code == 0, r.output
        assert len(r.output) > 50

    finally:
        if backup is not None:
            if conv.exists():
                conv.unlink()
            backup.rename(conv)
