"""THE strict-context contract.

This module is the only place that assembles prompts for Anthropic
generation calls. Everything depends on the system block being:

  1. Byte-stable across calls (so prompt caching works), and
  2. Faithful to the strict-context contract (so the synthetic data is
     actually grounded in the pillars rather than the frontier model's
     pretraining memory).

The snapshot test in tests/unit/test_prompt_builder.py is the alarm
bell for accidental regressions of (2). DO NOT WEAKEN THE LANGUAGE
WITHOUT UPDATING THE TEST.
"""

from __future__ import annotations

import importlib

from rosetta_bone.common.types import Chunk, Pillar

MIN_CHUNK_CHARS = 100

_DEFAULT_PERSONA_MODULE = "rosetta_bone.storyteller.sft.persona"


def _load_persona(module: str | None) -> str:
    """Import the configured persona module and return its PERSONA string.

    `None` resolves to the adult persona so callers that pre-date config
    routing (and the test suite) keep working unchanged.
    """
    return importlib.import_module(module or _DEFAULT_PERSONA_MODULE).PERSONA


_CONTRACT = """\
You write one (instruction, story) pair for fine-tuning a dog-POV
storyteller model.

The `instruction` MUST be a short user-style prompt (e.g., "Write a diary
entry about a trip to the vet from the dog's point of view.").

The `story` MUST be written from the dog's first-person sensory POV,
foregrounding scent, sound, and pheromonal cues over visual detail.

Ground the story strictly in the source material below. Do NOT invent
new science. Sensory mechanisms (volatile organic compounds, scent
plumes, vomeronasal cues, frequency-shifted hearing) MUST be drawn ONLY
from <science>. Voice and sentence rhythm MUST echo <style>.
Stimulus-to-reaction patterns MUST be plausible per <behavior>.

Return JSON only, with this shape:

  {"instruction": "...", "story": "..."}
"""


def build_system_block(
    chunks: dict[Pillar, Chunk],
    *,
    persona_module: str | None = None,
) -> str:
    for pillar, chunk in chunks.items():
        if len(chunk.text) < MIN_CHUNK_CHARS:
            raise ValueError(
                f"Chunk for pillar {pillar.value} is too short "
                f"({len(chunk.text)} < {MIN_CHUNK_CHARS} chars). "
                "Strict-context contract requires substantive grounding."
            )
    sci = chunks[Pillar.SCIENCE].text
    sty = chunks[Pillar.STYLE].text
    beh = chunks[Pillar.BEHAVIOR].text
    persona = _load_persona(persona_module)
    return (
        f"<persona>\n{persona}\n</persona>\n\n"
        f"<contract>\n{_CONTRACT}\n</contract>\n\n"
        f"<science>\n{sci}\n</science>\n\n"
        f"<style>\n{sty}\n</style>\n\n"
        f"<behavior>\n{beh}\n</behavior>\n"
    )


def build_messages(
    chunks: dict[Pillar, Chunk],
    *,
    stimulus: str,
    angle: str,
    form: str,
    variation: int,
    persona_module: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the system+user messages for one Anthropic request.

    `stimulus` is the user-facing scene name; `angle` is the specific
    behavioral/emotional slice that drove FAISS retrieval for this
    request. Both are surfaced to Claude so it can write a story that
    is recognizably *this* version of the scene (not just any version).
    """
    system = build_system_block(chunks, persona_module=persona_module)
    user = (
        f'Stimulus: "{stimulus}".\n'
        f'Angle: "{angle}".\n'
        f"Form: {form}.\n"
        f"Variation index: {variation}.\n"
        f"Return JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
