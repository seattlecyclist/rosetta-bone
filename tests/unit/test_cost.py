from rosetta_bone.storyteller.sft.cost import (
    PRICE_TABLE,
    Usage,
    estimate_cost_usd,
    sum_usage,
)


def test_price_table_has_sonnet_and_opus():
    assert "claude-sonnet-4-6" in PRICE_TABLE
    assert "claude-opus-4-7" in PRICE_TABLE


def test_estimate_cost_basic():
    u = Usage(input_tokens=1_000_000, output_tokens=0,
              cache_read_input_tokens=0, cache_creation_input_tokens=0)
    cost = estimate_cost_usd(u, model="claude-sonnet-4-6")
    assert abs(cost - PRICE_TABLE["claude-sonnet-4-6"]["input"]) < 1e-6


def test_sum_usage():
    a = Usage(1, 2, 3, 4)
    b = Usage(10, 20, 30, 40)
    s = sum_usage([a, b])
    assert s.input_tokens == 11
    assert s.cache_creation_input_tokens == 44
