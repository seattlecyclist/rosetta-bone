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

from rosetta_bone.common.types import Chunk, Pillar
from rosetta_bone.storyteller.sft.persona import PERSONA

MIN_CHUNK_CHARS = 100


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


def build_system_block(chunks: dict[Pillar, Chunk]) -> str:
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
    return (
        f"<persona>\n{PERSONA}\n</persona>\n\n"
        f"<contract>\n{_CONTRACT}\n</contract>\n\n"
        f"<science>\n{sci}\n</science>\n\n"
        f"<style>\n{sty}\n</style>\n\n"
        f"<behavior>\n{beh}\n</behavior>\n"
    )


def build_messages(
    chunks: dict[Pillar, Chunk],
    *,
    stimulus: str,
    form: str,
    variation: int,
) -> list[dict[str, str]]:
    system = build_system_block(chunks)
    user = (
        f'Stimulus: "{stimulus}".\n'
        f"Form: {form}.\n"
        f"Variation index: {variation}.\n"
        f"Return JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
