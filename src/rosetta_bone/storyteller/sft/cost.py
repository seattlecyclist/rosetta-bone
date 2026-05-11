"""Token & dollar accounting from Anthropic usage objects.

PRICE_TABLE is per-million-token in USD. Update it when Anthropic
changes prices. Batch API discount is 50% — apply via batch=True.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

PRICE_TABLE: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_creation": 3.75,
    },
    "claude-opus-4-7": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_creation": 18.75,
    },
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


def estimate_cost_usd(u: Usage, *, model: str, batch: bool = False) -> float:
    rates = PRICE_TABLE[model]
    cost = (
        u.input_tokens * rates["input"]
        + u.output_tokens * rates["output"]
        + u.cache_read_input_tokens * rates["cache_read"]
        + u.cache_creation_input_tokens * rates["cache_creation"]
    ) / 1_000_000
    return cost * (0.5 if batch else 1.0)


def sum_usage(items: Iterable[Usage]) -> Usage:
    inp = out = cr = cc = 0
    for u in items:
        inp += u.input_tokens
        out += u.output_tokens
        cr += u.cache_read_input_tokens
        cc += u.cache_creation_input_tokens
    return Usage(inp, out, cr, cc)
