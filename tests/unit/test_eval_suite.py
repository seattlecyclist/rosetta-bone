"""Tests for the frozen eval-suite module + its CLI surface."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from rosetta_bone.storyteller.cli import app
from rosetta_bone.storyteller.eval_suite import (
    EvalPrompt,
    compare_evals,
    eval_output_path,
    eval_set_sha,
    load_prompts,
)

runner = CliRunner()


def test_load_default_eval_prompts():
    prompts = load_prompts(Path("config/eval_prompts.yaml"))
    assert len(prompts) >= 5
    assert all(isinstance(p, EvalPrompt) for p in prompts)
    # Mix of categories so we test both fidelity and generalization.
    cats = {p.category for p in prompts}
    assert "trained" in cats and "novel" in cats


def test_eval_set_sha_is_stable_across_order_preserving_writes(tmp_path: Path):
    p = tmp_path / "set.yaml"
    p.write_text(yaml.safe_dump([
        {"prompt": "a", "form": "diary", "category": "trained"},
        {"prompt": "b", "form": "vignette", "category": "novel"},
    ]))
    a = eval_set_sha(load_prompts(p))
    b = eval_set_sha(load_prompts(p))
    assert a == b


def test_eval_set_sha_changes_when_prompts_change(tmp_path: Path):
    p1 = tmp_path / "v1.yaml"
    p2 = tmp_path / "v2.yaml"
    p1.write_text(yaml.safe_dump([{"prompt": "a", "form": "diary", "category": "trained"}]))
    p2.write_text(yaml.safe_dump([{"prompt": "b", "form": "diary", "category": "trained"}]))
    assert eval_set_sha(load_prompts(p1)) != eval_set_sha(load_prompts(p2))


def test_eval_output_path_uses_sha(tmp_path: Path):
    prompts = [EvalPrompt(prompt="x")]
    out = eval_output_path(tmp_path, prompts)
    sha = eval_set_sha(prompts)
    assert out.name == f"eval-{sha}.json"
    assert out.parent == tmp_path


def test_compare_evals_renders_both_stories():
    a = {
        "adapter_dir": "/a",
        "eval_set_sha": "abc1234567",
        "results": [
            {"prompt": "vet visit", "form": "diary", "category": "trained",
             "story": "A_STORY_FOR_VET"},
        ],
    }
    b = {
        "adapter_dir": "/b",
        "eval_set_sha": "abc1234567",
        "results": [
            {"prompt": "vet visit", "form": "diary", "category": "trained",
             "story": "B_STORY_FOR_VET"},
        ],
    }
    out = compare_evals(a, b)
    assert "/a" in out and "/b" in out
    assert "A_STORY_FOR_VET" in out
    assert "B_STORY_FOR_VET" in out
    assert "vet visit" in out


def test_compare_evals_warns_on_sha_mismatch():
    a = {"adapter_dir": "/a", "eval_set_sha": "abc1234567", "results": []}
    b = {"adapter_dir": "/b", "eval_set_sha": "deadbeef12", "results": []}
    out = compare_evals(a, b)
    assert "eval_set_sha differs" in out


def test_eval_cli_help_documents_adapter_and_force():
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    assert "--adapter" in r.output
    assert "--force" in r.output


def test_eval_compare_cli_help():
    r = runner.invoke(app, ["eval-compare", "--help"])
    assert r.exit_code == 0


def test_generate_cli_help_documents_adapter_flag():
    r = runner.invoke(app, ["generate", "--help"])
    assert r.exit_code == 0
    assert "--adapter" in r.output
