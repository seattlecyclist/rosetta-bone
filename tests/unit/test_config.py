from pathlib import Path

from rosetta_bone.common.config import Config, load_config


def test_load_default_config():
    cfg = load_config(Path("config/default.toml"))
    assert isinstance(cfg, Config)
    assert cfg.paths.data_dir.name == "data"
    assert cfg.sft.max_requests_per_run == 1000
    assert cfg.sft.requests_per_minute == 50
    assert cfg.train.base_model == "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"


def test_load_config_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("""
[paths]
data_dir = "/tmp/foo"
raw_dir = "/tmp/foo/raw"
chunks_dir = "/tmp/foo/chunks"
embeddings_dir = "/tmp/foo/embeddings"
sft_dir = "/tmp/foo/sft"
adapter_dir = "/tmp/foo/adapters/x"

[retrieval]
embedding_model = "x"
similarity_threshold = 0.3

[sft]
model = "claude-sonnet-4-6"
max_requests_per_run = 50
requests_per_minute = 10
batch_size_max = 100

[train]
base_model = "x"
rank = 4
alpha = 8.0
iters = 10
batch_size = 1
learning_rate = 0.001
target_modules = ["q_proj"]

[infer]
temperature = 0.5
top_p = 0.9
max_tokens = 100
repetition_penalty = 1.0
""")
    cfg = load_config(p)
    assert cfg.sft.max_requests_per_run == 50
