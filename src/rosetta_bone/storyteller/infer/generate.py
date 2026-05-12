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
    adapter_override: Path | None = None,
    config_path: Path = Path("config/default.toml"),
) -> str:
    """Run inference against the fine-tuned model.

    mlx-lm's sampling API moved away from raw temp=/top_p=/
    repetition_penalty= kwargs in recent releases. Sampling is now
    configured via `sampler` (a callable built by make_sampler) and
    `logits_processors` (built by make_logits_processors).

    `adapter_override`: if set, load this specific adapter directory
    (typically a timestamped versioned dir) instead of resolving the
    'latest' symlink under cfg.paths.adapter_dir. Used by `eval` and
    by `generate --adapter <ts>` to compare specific training runs.
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    cfg = load_config(config_path)
    adapter_dir = adapter_override if adapter_override is not None else cfg.paths.adapter_dir
    model, tokenizer = load(cfg.train.base_model, adapter_dir)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": _format_prompt(stimulus, form)}],
        add_generation_prompt=True,
        tokenize=False,
    )
    sampler = make_sampler(
        temp=temperature if temperature is not None else cfg.infer.temperature,
        top_p=top_p if top_p is not None else cfg.infer.top_p,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=cfg.infer.repetition_penalty,
    )
    return mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens or cfg.infer.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
