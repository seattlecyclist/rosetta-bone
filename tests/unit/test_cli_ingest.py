from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for sub in ["ingest", "chunk"]:
        assert sub in r.output
