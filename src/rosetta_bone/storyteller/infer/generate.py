"""Public inference entry point."""

from __future__ import annotations

from pathlib import Path

from rosetta_bone.common.config import load_config
from rosetta_bone.storyteller.infer.model import load


def _format_prompt(stimulus: str, form: str = "diary") -> str:
    return (
        f"Write a {form} entry from a dog's first-person sensory point of view "
        f"about the following stimulus: {stimulus}."
    )


def generate(
    stimulus: str,
    *,
    form: str = "diary",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    config_path: Path = Path("config/default.toml"),
) -> str:
    from mlx_lm import generate as mlx_generate

    cfg = load_config(config_path)
    model, tokenizer = load(cfg.train.base_model, cfg.paths.adapter_dir)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": _format_prompt(stimulus, form)}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens or cfg.infer.max_tokens,
        temp=temperature or cfg.infer.temperature,
        top_p=top_p or cfg.infer.top_p,
        repetition_penalty=cfg.infer.repetition_penalty,
        verbose=False,
    )
