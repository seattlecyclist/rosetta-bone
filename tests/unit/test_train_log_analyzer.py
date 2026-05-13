"""Tests for the mlx-lm training log analyzer.

Uses static fixture log snippets copied verbatim from real mlx-lm
output (with line wrapping preserved exactly as the binary emits).
"""

from pathlib import Path

from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app
from rosetta_bone.storyteller.train.log_analyzer import (
    format_report,
    overfit_verdict,
    parse_log,
)

runner = CliRunner()


_FIXTURE = """\
Loading datasets
Iter 1: Val loss 2.439, Val took 5.012s
Iter 10: Train loss 1.834, Learning Rate 1.000e-05, It/sec 0.130, Tokens/sec 270.000, Trained Tokens 21000, Peak mem 9.500 GB
Iter 20: Train loss 1.612, Learning Rate 1.000e-05, It/sec 0.135, Tokens/sec 280.000, Trained Tokens 42000, Peak mem 9.600 GB
Iter 200: Val loss 1.556, Val took 5.000s
Iter 200: Train loss 1.105, Learning Rate 1.000e-05, It/sec 0.140, Tokens/sec 290.000, Trained Tokens 420000, Peak mem 9.700 GB
Iter 400: Val loss 1.630, Val took 5.000s
Iter 400: Train loss 0.832, Learning Rate 1.000e-05, It/sec 0.142, Tokens/sec 295.000, Trained Tokens 840000, Peak mem 9.700 GB
Iter 600: Val loss 1.750, Val took 5.000s
Iter 600: Train loss 0.610, Learning Rate 1.000e-05, It/sec 0.140, Tokens/sec 290.000, Trained Tokens 1260000, Peak mem 9.700 GB
Iter 800: Val loss 1.955, Val took 5.000s
Iter 800: Train loss 0.420, Learning Rate 1.000e-05, It/sec 0.138, Tokens/sec 285.000, Trained Tokens 1680000, Peak mem 9.700 GB
Iter 800: Saved adapter weights to .../0000800_adapters.safetensors.
Iter 1000: Train loss 0.260, Learning Rate 1.000e-05, It/sec 0.150, Tokens/sec 310.000, Trained Tokens 2100000, Peak mem 9.700 GB
Iter 1000: Val loss 2.172, Val took 5.000s
Iter 1000: Train loss 0.135, Learning Rate 1.000e-05, It/sec 0.150, Tokens/sec 310.000, Trained Tokens 2100000, Peak mem 9.700 GB
"""


def test_parse_log_captures_train_and_validation_series(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)

    # Train loss reports at iters 10, 20, 200, 400, 600, 800, 1000 (x2)
    assert rep.n_train_reports == 8
    train_iters = [it for it, _ in rep.train_loss_series]
    assert 10 in train_iters and 1000 in train_iters

    # Validation reports at iter 1, 200, 400, 600, 800, 1000
    assert rep.n_validation_reports == 6
    assert rep.validation_loss_series[0] == (1, 2.439)
    assert rep.validation_loss_series[-1] == (1000, 2.172)


def test_parse_log_tracks_peak_memory_and_trained_tokens(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)
    assert rep.peak_memory_gb == 9.7
    assert rep.trained_tokens == 2_100_000


def test_parse_log_handles_missing_file(tmp_path: Path):
    rep = parse_log(tmp_path / "nope.log")
    assert rep.n_train_reports == 0
    assert rep.n_validation_reports == 0


def test_overfit_verdict_calls_out_deep_memorization(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)
    v = overfit_verdict(rep)
    # Final train loss 0.135, validation min was 1.556 climbing to 2.172
    # That hits the deep-memorization branch.
    assert "memorization" in v.lower()


def test_overfit_verdict_underfit_when_final_train_high(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text("""\
Iter 1: Val loss 2.0, Val took 5s
Iter 10: Train loss 1.500, Learning Rate 1.000e-05, It/sec 0.1, Tokens/sec 100, Trained Tokens 1000, Peak mem 1.0 GB
Iter 200: Val loss 1.9, Val took 5s
Iter 200: Train loss 0.900, Learning Rate 1.000e-05, It/sec 0.1, Tokens/sec 100, Trained Tokens 20000, Peak mem 1.0 GB
""")
    rep = parse_log(p)
    v = overfit_verdict(rep)
    assert "underfit" in v.lower()


def test_format_report_uses_validation_loss_not_val_loss(tmp_path: Path):
    """mlx-lm says 'Val loss'; our report MUST say 'validation loss'.

    Mike asked for the abbreviation to never appear in our renderings.
    See memory/terminology.md.
    """
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)
    out = format_report(rep)
    assert "VALIDATION LOSS" in out
    # We allow "validation" in the body; we never allow the abbreviation
    # standing alone. Spot-check the literal "Val loss" never appears
    # in the rendered output even though it's all over the input.
    assert "Val loss" not in out


def test_format_report_includes_throughput_and_train_loss_summary(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)
    out = format_report(rep)
    assert "THROUGHPUT" in out
    assert "iterations per second" in out
    assert "TRAIN LOSS" in out
    assert "min observed" in out
    assert "VERDICT" in out


def test_format_report_shows_full_validation_series(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    rep = parse_log(p)
    out = format_report(rep)
    # Every validation iter must appear in the rendered series
    for iter_n in (1, 200, 400, 600, 800, 1000):
        assert f"iter {iter_n:>5d}" in out


def test_cli_train_inspect_help():
    r = runner.invoke(app, ["train-inspect", "--help"])
    assert r.exit_code == 0
    assert "--adapter" in r.output
    assert "--log-file" in r.output


def test_cli_train_inspect_with_log_file(tmp_path: Path):
    p = tmp_path / "train.log"
    p.write_text(_FIXTURE)
    r = runner.invoke(app, ["train-inspect", "--log-file", str(p)])
    assert r.exit_code == 0
    assert "VALIDATION LOSS" in r.output
    assert "Val loss" not in r.output
