"""Pre-training inspection of the merged SFT corpus.

Run after `sft merge`, before `train`. Joins the raw batch results
(which carry custom_id -> stimulus+angle attribution) with the merged
train+valid (which carries survivorship after dedup) to produce:

  - Overall counts: raw, errored, invalid-JSON, kept-after-dedup
  - Per-stimulus pair counts + dedup rates
  - Per-(stimulus, angle) breakdown so you can spot which angles
    aren't earning their keep
  - Story token-length distribution (p10/p50/p90/max)
  - Persona-violation flags: substring scan for the markers the
    persona explicitly forbids ("olfactory", "vessel", "I contemplated",
    "the way a smell", ...)

The point is to catch a bad pilot before spending GPU time on it.
"""

from __future__ import annotations

import importlib
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosetta_bone.common.chunking import count_tokens
from rosetta_bone.common.jsonl import iter_jsonl
from rosetta_bone.common.logging import get_logger
from rosetta_bone.storyteller.sft.generate import _slug
from rosetta_bone.storyteller.sft.merge import _hash_instr, parse_assistant_json
from rosetta_bone.storyteller.sft.persona import PERSONA_VIOLATIONS as _DEFAULT_VIOLATIONS
from rosetta_bone.storyteller.sft.stimuli import Stimulus, load_stimuli

_log = get_logger(__name__)


def _load_persona_violations(module: str) -> tuple[str, ...]:
    """Import the configured persona module and return its PERSONA_VIOLATIONS."""
    return importlib.import_module(module).PERSONA_VIOLATIONS


@dataclass
class _StimulusAccumulator:
    stimulus: str
    n_generated: int = 0
    n_kept: int = 0
    angle_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    story_token_lengths: list[int] = field(default_factory=list)
    persona_violations: int = 0


def _custom_id_parts(custom_id: str) -> tuple[str, str, int, int] | None:
    """Parse `{phase}__{slug}__a{angle_idx}__v{var_idx}`.

    Uses `rsplit("__", 3)` so a phase tag may itself contain `__` if a
    future workflow needs it. Returns None on a malformed id.
    """
    parts = custom_id.rsplit("__", 3)
    if len(parts) != 4:
        return None
    phase, slug, angle_marker, var_marker = parts
    if not angle_marker.startswith("a") or not var_marker.startswith("v"):
        return None
    try:
        return phase, slug, int(angle_marker[1:]), int(var_marker[1:])
    except ValueError:
        return None


def _build_attribution_map(
    stimuli: Iterable[Stimulus],
) -> dict[tuple[str, int], tuple[str, str, str]]:
    """Map (slug, angle_idx) -> (stimulus_prompt, angle_text, form)."""
    out: dict[tuple[str, int], tuple[str, str, str]] = {}
    for s in stimuli:
        slug = _slug(s.prompt)
        for angle_idx, angle in enumerate(s.embed_queries):
            out[(slug, angle_idx)] = (s.prompt, angle, s.form)
    return out


def _count_persona_violations(
    text: str,
    violations: tuple[str, ...] = _DEFAULT_VIOLATIONS,
) -> int:
    lowered = text.lower()
    return sum(1 for marker in violations if marker.lower() in lowered)


def _kept_instruction_hashes(*jsonl_paths: Path) -> set[str]:
    """Load every train/valid record, return SHA-1s of the user (instruction) turns."""
    hashes: set[str] = set()
    for path in jsonl_paths:
        for row in iter_jsonl(path):
            user_msg = next(
                (m for m in row.get("messages", []) if m.get("role") == "user"),
                None,
            )
            if user_msg:
                hashes.add(_hash_instr(user_msg["content"]))
    return hashes


def count_corpus_tokens(*jsonl_paths: Path) -> dict[str, int]:
    """Sum user + assistant tokens across one or more chat-format JSONL files.

    Returns a dict with `user`, `assistant`, `total`, and `n_pairs`.
    Used to express corpus size in tokens (rather than just pair count)
    so different pilots can be compared on a like-for-like axis.
    """
    user_total = 0
    assistant_total = 0
    n_pairs = 0
    for path in jsonl_paths:
        for row in iter_jsonl(path):
            msgs = row.get("messages", [])
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            assist_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user_msg:
                user_total += count_tokens(user_msg["content"])
            if assist_msg:
                assistant_total += count_tokens(assist_msg["content"])
            n_pairs += 1
    return {
        "n_pairs": n_pairs,
        "user": user_total,
        "assistant": assistant_total,
        "total": user_total + assistant_total,
    }


def _quantiles(xs: list[int]) -> dict[str, int]:
    if not xs:
        return {"p10": 0, "p50": 0, "p90": 0, "max": 0}
    s = sorted(xs)
    n = len(s)
    return {
        "p10": s[max(0, int(n * 0.1) - 1)] if n > 1 else s[0],
        "p50": int(statistics.median(s)),
        "p90": s[min(n - 1, int(n * 0.9))] if n > 1 else s[0],
        "max": s[-1],
    }


def compute_stats(
    *,
    batches_dir: Path,
    train_path: Path,
    valid_path: Path,
    stimuli_path: Path,
    persona_module: str | None = None,
) -> dict[str, Any]:
    """Compute pre-training stats for the merged SFT corpus.

    `persona_module` selects which persona's PERSONA_VIOLATIONS list to
    scan stories against. When None, falls back to the default (adult)
    persona's violations — preserving existing behavior for callers
    that haven't been updated to pass cfg.persona.module.
    """
    violations = (
        _load_persona_violations(persona_module)
        if persona_module
        else _DEFAULT_VIOLATIONS
    )
    stimuli = load_stimuli(stimuli_path)
    attribution = _build_attribution_map(stimuli)
    kept_hashes = _kept_instruction_hashes(train_path, valid_path)

    per_stim: dict[str, _StimulusAccumulator] = {}
    n_raw = 0
    n_errored = 0
    n_invalid_json = 0
    n_unattributed = 0
    all_story_tokens: list[int] = []
    total_persona_violations = 0
    # Walk in the same sorted-glob order merge.py uses so we can credit
    # the FIRST occurrence of each unique instruction as kept; later
    # duplicates count toward n_generated (visible in dedup rates) but
    # are not counted toward n_kept.
    seen_hashes: set[str] = set()

    for batch_file in sorted(batches_dir.glob("*.jsonl")):
        for row in iter_jsonl(batch_file):
            n_raw += 1
            if row.get("type") != "succeeded":
                n_errored += 1
                continue
            parsed = parse_assistant_json(row.get("text", ""))
            if parsed is None:
                n_invalid_json += 1
                continue
            cid_parts = _custom_id_parts(row.get("custom_id", ""))
            if cid_parts is None:
                n_unattributed += 1
                continue
            _, slug, angle_idx, _ = cid_parts
            attr = attribution.get((slug, angle_idx))
            if attr is None:
                n_unattributed += 1
                continue
            stimulus, angle, _form = attr

            ss = per_stim.setdefault(
                stimulus, _StimulusAccumulator(stimulus=stimulus),
            )
            ss.n_generated += 1
            ab = ss.angle_breakdown.setdefault(
                angle, {"n_generated": 0, "n_kept": 0},
            )
            ab["n_generated"] += 1

            h = _hash_instr(parsed["instruction"])
            first_occurrence = h not in seen_hashes
            seen_hashes.add(h)
            kept_by_merge = first_occurrence and h in kept_hashes
            if kept_by_merge:
                ss.n_kept += 1
                ab["n_kept"] += 1
                tokens = count_tokens(parsed["story"])
                ss.story_token_lengths.append(tokens)
                all_story_tokens.append(tokens)
                n_violations = _count_persona_violations(parsed["story"], violations)
                ss.persona_violations += n_violations
                total_persona_violations += n_violations

    n_generated_valid = sum(s.n_generated for s in per_stim.values())
    n_kept = sum(s.n_kept for s in per_stim.values())

    train_tokens = count_corpus_tokens(train_path)
    valid_tokens = count_corpus_tokens(valid_path)

    return {
        "summary": {
            "n_raw": n_raw,
            "n_errored": n_errored,
            "n_invalid_json": n_invalid_json,
            "n_unattributed": n_unattributed,
            "n_generated_valid": n_generated_valid,
            "n_kept": n_kept,
            "kept_fraction": (n_kept / n_generated_valid) if n_generated_valid else 0.0,
            "total_persona_violations": total_persona_violations,
        },
        "corpus_tokens": {
            "train": train_tokens,
            "valid": valid_tokens,
            "train_plus_valid_assistant": train_tokens["assistant"] + valid_tokens["assistant"],
        },
        "story_tokens": _quantiles(all_story_tokens),
        "per_stimulus": [
            {
                "stimulus": s.stimulus,
                "n_generated": s.n_generated,
                "n_kept": s.n_kept,
                "kept_fraction": (s.n_kept / s.n_generated) if s.n_generated else 0.0,
                "story_tokens_p50": (
                    int(statistics.median(s.story_token_lengths))
                    if s.story_token_lengths else 0
                ),
                "persona_violations": s.persona_violations,
                "angles": [
                    {
                        "angle": angle,
                        "n_generated": ab["n_generated"],
                        "n_kept": ab["n_kept"],
                        "kept_fraction": (
                            (ab["n_kept"] / ab["n_generated"])
                            if ab["n_generated"] else 0.0
                        ),
                    }
                    for angle, ab in s.angle_breakdown.items()
                ],
            }
            for s in sorted(per_stim.values(), key=lambda x: x.stimulus)
        ],
    }


def format_stats_table(stats: dict[str, Any]) -> str:
    """Render the stats dict as a terminal-friendly text report."""
    s = stats["summary"]
    t = stats["story_tokens"]
    lines: list[str] = []

    lines.append("SUMMARY")
    lines.append(f"  raw batch results:    {s['n_raw']}")
    lines.append(f"  errored:              {s['n_errored']}")
    lines.append(f"  invalid JSON:         {s['n_invalid_json']}")
    lines.append(f"  unattributed:         {s['n_unattributed']}")
    lines.append(f"  generated + valid:    {s['n_generated_valid']}")
    lines.append(
        f"  kept after dedup:     {s['n_kept']}  "
        f"({int(s['kept_fraction'] * 100)}%)"
    )
    lines.append(f"  persona violations:   {s['total_persona_violations']}")
    lines.append("")

    lines.append("STORY LENGTH (tokens, kept pairs only)")
    lines.append(
        f"  p10={t['p10']}  p50={t['p50']}  p90={t['p90']}  max={t['max']}"
    )
    lines.append("")

    ct = stats.get("corpus_tokens", {})
    if ct:
        tr = ct.get("train", {})
        vl = ct.get("valid", {})
        lines.append("CORPUS SIZE  (assistant text is the model's training target)")
        lines.append(
            f"  train: {tr.get('n_pairs', 0):>4d} pairs   "
            f"assistant={tr.get('assistant', 0):>7,d} tok   "
            f"user={tr.get('user', 0):>6,d} tok"
        )
        lines.append(
            f"  valid: {vl.get('n_pairs', 0):>4d} pairs   "
            f"assistant={vl.get('assistant', 0):>7,d} tok   "
            f"user={vl.get('user', 0):>6,d} tok"
        )
        lines.append("")

    lines.append("PER-STIMULUS")
    lines.append(
        f"  {'stimulus':40s} {'gen':>4s} {'kept':>5s} "
        f"{'pct':>4s} {'p50tok':>7s} {'viol':>4s}"
    )
    for ps in stats["per_stimulus"]:
        pct = f"{int(ps['kept_fraction'] * 100)}%"
        lines.append(
            f"  {ps['stimulus'][:40]:40s} "
            f"{ps['n_generated']:>4d} {ps['n_kept']:>5d} "
            f"{pct:>4s} {ps['story_tokens_p50']:>7d} "
            f"{ps['persona_violations']:>4d}"
        )
    lines.append("")

    lines.append(
        "PER-ANGLE  (low kept% = angle producing duplicates; redesign or drop)"
    )
    for ps in stats["per_stimulus"]:
        for a in ps["angles"]:
            pct = f"{int(a['kept_fraction'] * 100)}%"
            lines.append(
                f"  [{pct:>4s} kept] {ps['stimulus'][:30]:30s} "
                f":: {a['angle'][:55]}"
            )
    return "\n".join(lines)
