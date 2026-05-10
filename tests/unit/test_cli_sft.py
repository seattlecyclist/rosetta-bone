from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app

runner = CliRunner()


def test_sft_help_lists_subcommands():
    r = runner.invoke(app, ["sft", "--help"])
    assert r.exit_code == 0
    for sub in ["generate", "poll", "merge"]:
        assert sub in r.output


def test_sft_generate_rejects_oversize_count(tmp_path):
    r = runner.invoke(app, ["sft", "generate",
                            "--count", "5000",
                            "--max-requests", "1000",
                            "--phase", "pilot"])
    assert r.exit_code != 0
    assert "cap" in r.output.lower()
