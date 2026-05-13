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


def test_train_invokes_subprocess_run(tmp_path: Path):
    with patch("subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = train(
            base_model="m",
            train_data=tmp_path / "train.jsonl",
            valid_data=tmp_path / "valid.jsonl",
            adapter_dir=tmp_path / "adapter",
            rank=4, alpha=8.0, iters=10, batch_size=1, learning_rate=1e-4,
        )
        assert result.returncode == 0
        assert run.called
