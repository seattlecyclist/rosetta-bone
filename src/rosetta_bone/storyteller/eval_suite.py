"""Frozen evaluation suite for comparing trained adapters.

Reads a fixed list of prompts from `config/eval_prompts.yaml`, runs each
through `generate()` with a specified adapter, and writes the results
into the adapter's own directory:

    data/adapters/<...>/<timestamp>/eval-<sha>.json

The `<sha>` is a stable hash of the prompt set. Re-running with the same
prompts is idempotent (skip if file exists); changing the prompts
produces a new file alongside the old one. Old eval files stay on disk
so historical comparison is always possible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from rosetta_bone.common.logging import get_logger

_log = get_logger(__name__)


class EvalPrompt(BaseModel):
    prompt: str
    form: str = "diary"
    category: str = "trained"


def load_prompts(path: Path) -> list[EvalPrompt]:
    raw = yaml.safe_load(path.read_text())
    return [EvalPrompt.model_validate(r) for r in raw]


def eval_set_sha(prompts: Iterable[EvalPrompt]) -> str:
    """Stable 10-char SHA-1 of the eval set. Changes when prompts change."""
    norm = json.dumps([p.model_dump() for p in prompts], sort_keys=True)
    return hashlib.sha1(norm.encode()).hexdigest()[:10]


def eval_output_path(adapter_dir: Path, prompts: Iterable[EvalPrompt]) -> Path:
    return adapter_dir / f"eval-{eval_set_sha(prompts)}.json"


def run_eval(
    *,
    adapter_dir: Path,
    base_model: str,
    prompts: list[EvalPrompt],
    max_tokens: int | None = None,
    force: bool = False,
) -> Path:
    """Run all prompts against `adapter_dir`; write results into it.

    Idempotent: if the eval file already exists and `force` is False,
    returns the existing path without regenerating. Saves wall time
    and avoids overwriting prior results that may be referenced.
    """
    from rosetta_bone.storyteller.infer.generate import generate

    out_path = eval_output_path(adapter_dir, prompts)
    if out_path.exists() and not force:
        _log.info("eval_skip_existing", path=str(out_path))
        return out_path

    started = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        _log.info("eval_generate", idx=i + 1, n=len(prompts), prompt=p.prompt)
        text = generate(
            p.prompt,
            form=p.form,
            max_tokens=max_tokens,
            adapter_override=adapter_dir,
        )
        results.append({
            "prompt": p.prompt,
            "form": p.form,
            "category": p.category,
            "story": text,
        })

    payload = {
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "eval_set_sha": eval_set_sha(prompts),
        "n_prompts": len(prompts),
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    _log.info("eval_done", n=len(results), output=str(out_path))
    return out_path


def compare_evals(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Render two eval payloads side-by-side as a single string.

    Output groups by prompt and renders A then B for each. The reader
    eyeballs the diff.
    """
    a_results = {r["prompt"]: r for r in a["results"]}
    b_results = {r["prompt"]: r for r in b["results"]}
    prompts_in_order = list(a_results) + [p for p in b_results if p not in a_results]

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"A: {a['adapter_dir']}")
    lines.append(f"B: {b['adapter_dir']}")
    if a.get("eval_set_sha") != b.get("eval_set_sha"):
        lines.append(
            f"⚠ eval_set_sha differs ({a.get('eval_set_sha')} vs {b.get('eval_set_sha')}); "
            "the two runs evaluated against different prompt sets."
        )
    lines.append("=" * 80)

    for prompt in prompts_in_order:
        ra = a_results.get(prompt)
        rb = b_results.get(prompt)
        category = (ra or rb or {}).get("category", "?")
        form = (ra or rb or {}).get("form", "?")
        lines.append("")
        lines.append(f"### [{category} · {form}]  {prompt}")
        lines.append("--- A " + "-" * 74)
        lines.append(ra["story"] if ra else "(not present in A)")
        lines.append("--- B " + "-" * 74)
        lines.append(rb["story"] if rb else "(not present in B)")

    return "\n".join(lines)
