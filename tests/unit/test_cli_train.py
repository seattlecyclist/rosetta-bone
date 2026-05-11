from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app

runner = CliRunner()


def test_train_help():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0
    assert "iters" in r.output
