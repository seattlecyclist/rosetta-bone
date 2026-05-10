from rosetta_bone.storyteller.train.eval import parse_perplexity


def test_parse_perplexity_from_mlx_output():
    sample = """\
Iter 100: train loss 1.234
Iter 200: val loss 1.123
Test loss 1.045, Test ppl 2.84
"""
    ppl = parse_perplexity(sample)
    assert ppl == 2.84


def test_parse_perplexity_returns_none_when_missing():
    assert parse_perplexity("no relevant output") is None
