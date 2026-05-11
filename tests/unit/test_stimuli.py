from pathlib import Path

from rosetta_bone.storyteller.sft.stimuli import (
    Stimulus,
    expand,
    load_stimuli,
)


def test_load_default_stimuli():
    stimuli = load_stimuli(Path("config/stimuli.yaml"))
    assert len(stimuli) >= 15
    assert all(isinstance(s, Stimulus) for s in stimuli)
    assert all(s.variations >= 1 for s in stimuli)


def test_expand():
    s = [
        Stimulus(prompt="vet visit", variations=3, form="diary"),
        Stimulus(prompt="mailman", variations=2, form="vignette"),
    ]
    pairs = list(expand(s))
    assert len(pairs) == 5
    assert pairs[0] == ("vet visit", 0, "diary")
    assert pairs[2] == ("vet visit", 2, "diary")
    assert pairs[3] == ("mailman", 0, "vignette")
