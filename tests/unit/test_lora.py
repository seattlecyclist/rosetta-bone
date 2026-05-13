from pathlib import Path
from unittest.mock import MagicMock, patch

from rosetta_bone.storyteller.train.lora import build_train_argv, train


def test_build_train_argv_includes_required_flags(tmp_path: Path):
    argv = build_train_argv(
        base_model="mlx-community/foo",
        data_dir=tmp_path / "data" / "sft",
        adapter_dir=tmp_path / "adapter",
        rank=8, alpha=16.0, iters=200, batch_size=4, learning_rate=1e-5,
    )
    s = " ".join(argv)
    assert "mlx_lm.lora" in argv
    assert "--train" in argv
    assert "--model" in argv and "mlx-community/foo" in s
    assert "--iters" in argv and "200" in argv
    assert "--batch-size" in argv and "4" in argv
    assert "--adapter-path" in argv
    assert "--num-layers" in argv
    # mlx-lm has no --lora-layers flag; passing it would error with
    # "unrecognized arguments". Guard against accidental re-addition.
    assert "--lora-layers" not in argv
    # Memory + speed knobs. We keep --num-layers and --max-seq-length
    # tuned down; --grad-checkpoint was removed once peak memory was
    # measured at ~9.7 GB on 32 GB (huge headroom). Re-add it if any
    # future train run OOMs.
    assert "--grad-checkpoint" not in argv
    assert "--max-seq-length" in argv and "1024" in argv
    num_layers_idx = argv.index("--num-layers")
    assert argv[num_layers_idx + 1] == "8"


def test_train_invokes_subprocess_and_writes_log(tmp_path: Path):
    # train() now tees mlx-lm's stdout to both the parent terminal and
    # <adapter_dir>/train.log, so we mock subprocess.Popen and assert
    # the log file was written.
    fake_lines = [
        "Iter 10: Train loss 1.0, It/sec 0.1, Tokens/sec 100, "
        "Trained Tokens 1000, Peak mem 1.0 GB\n",
        "Iter 20: Train loss 0.9, It/sec 0.1, Tokens/sec 100, "
        "Trained Tokens 2000, Peak mem 1.0 GB\n",
    ]
    fake_proc = MagicMock()
    fake_proc.stdout = iter(fake_lines)
    fake_proc.wait.return_value = 0

    adapter_dir = tmp_path / "adapter"
    with patch("subprocess.Popen", return_value=fake_proc) as popen:
        result = train(
            base_model="m",
            train_data=tmp_path / "train.jsonl",
            valid_data=tmp_path / "valid.jsonl",
            adapter_dir=adapter_dir,
            rank=4, alpha=8.0, iters=10, batch_size=1, learning_rate=1e-4,
        )
        assert popen.called
        assert result.returncode == 0

    log_path = adapter_dir / "train.log"
    assert log_path.exists()
    assert "Iter 10: Train loss 1.0" in log_path.read_text()
