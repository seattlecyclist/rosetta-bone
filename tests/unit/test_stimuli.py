from pathlib import Path

import pytest

from rosetta_bone.storyteller.sft.stimuli import (
    Stimulus,
    expand,
    load_stimuli,
)


def test_load_default_stimuli():
    stimuli = load_stimuli(Path("config/stimuli.yaml"))
    assert len(stimuli) >= 15
    assert all(isinstance(s, Stimulus) for s in stimuli)
    # Every stimulus must declare at least one retrieval angle.
    assert all(len(s.embed_queries) >= 1 for s in stimuli)
    assert all(s.variations_per_query >= 1 for s in stimuli)
    # Curated default uses behavioral angles, not bare paraphrases.
    sample = stimuli[0]
    assert sample.embed_queries != [sample.prompt]


def test_expand_yields_five_tuples():
    s = [
        Stimulus(
            prompt="vet visit",
            embed_queries=["dog anxious at the vet", "dog excited for vet treat"],
            variations_per_query=2,
            form="diary",
        ),
        Stimulus(
            prompt="thunderstorm",
            embed_queries=["dog hiding from the noise"],
            variations_per_query=3,
            form="vignette",
            modality="hearing",
        ),
    ]
    pairs = list(expand(s))
    # 2 angles * 2 vars + 1 angle * 3 vars = 7
    assert len(pairs) == 7

    # Shape: (stimulus, embed_query, variation_idx, form, modality).
    # modality is None when unset on the Stimulus, otherwise the literal value.
    assert pairs[0] == ("vet visit", "dog anxious at the vet", 0, "diary", None)
    assert pairs[1] == ("vet visit", "dog anxious at the vet", 1, "diary", None)
    assert pairs[2] == ("vet visit", "dog excited for vet treat", 0, "diary", None)
    assert pairs[3] == ("vet visit", "dog excited for vet treat", 1, "diary", None)
    assert pairs[4] == ("thunderstorm", "dog hiding from the noise", 0, "vignette", "hearing")
    assert pairs[6] == ("thunderstorm", "dog hiding from the noise", 2, "vignette", "hearing")


def test_expand_variation_index_resets_per_angle():
    s = [Stimulus(
        prompt="x",
        embed_queries=["a", "b"],
        variations_per_query=3,
        form="diary",
    )]
    pairs = list(expand(s))
    indices_per_angle = {}
    for _, angle, var, _, _ in pairs:
        indices_per_angle.setdefault(angle, []).append(var)
    assert indices_per_angle == {"a": [0, 1, 2], "b": [0, 1, 2]}


def test_stimulus_rejects_invalid_modality():
    with pytest.raises(ValueError):
        Stimulus(
            prompt="x",
            embed_queries=["a"],
            variations_per_query=1,
            form="diary",
            modality="taste",  # not in {smell, hearing}
        )


def test_stimulus_rejects_empty_embed_queries():
    with pytest.raises(ValueError, match="at least one"):
        Stimulus(prompt="x", embed_queries=[], variations_per_query=1, form="diary")


def test_stimulus_rejects_whitespace_only_embed_query():
    with pytest.raises(ValueError, match="non-empty"):
        Stimulus(
            prompt="x",
            embed_queries=["good angle", "  "],
            variations_per_query=1,
            form="diary",
        )
