from __future__ import annotations

from rosetta_bone.storyteller.train.remote.keys import (
    KEY_LENGTH,
    adapter_key,
    adapter_prefix,
    dataset_prefix,
)


def _hp(**overrides) -> dict:
    base = {
        "rank": 8,
        "alpha": 16.0,
        "iters": 1000,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "target_modules": ["q_proj", "v_proj"],
    }
    base.update(overrides)
    return base


def test_key_is_deterministic_and_short():
    k = adapter_key(
        train_sha1="a" * 40,
        valid_sha1="b" * 40,
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        hyperparams=_hp(),
    )
    assert len(k) == KEY_LENGTH
    assert all(c in "0123456789abcdef" for c in k)


def test_key_invariant_under_hyperparam_dict_order():
    a = adapter_key(
        train_sha1="x", valid_sha1="y", base_model="m",
        hyperparams={"a": 1, "b": 2, "c": 3},
    )
    b = adapter_key(
        train_sha1="x", valid_sha1="y", base_model="m",
        hyperparams={"c": 3, "a": 1, "b": 2},
    )
    assert a == b


def test_key_changes_when_learning_rate_changes():
    a = adapter_key(
        train_sha1="x", valid_sha1="y", base_model="m",
        hyperparams=_hp(learning_rate=1e-5),
    )
    b = adapter_key(
        train_sha1="x", valid_sha1="y", base_model="m",
        hyperparams=_hp(learning_rate=2e-5),
    )
    assert a != b


def test_key_changes_when_data_changes():
    a = adapter_key(
        train_sha1="aaa", valid_sha1="bbb", base_model="m", hyperparams=_hp(),
    )
    b = adapter_key(
        train_sha1="aaa", valid_sha1="ccc", base_model="m", hyperparams=_hp(),
    )
    assert a != b


def test_key_changes_when_base_model_changes():
    a = adapter_key(
        train_sha1="x", valid_sha1="y",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        hyperparams=_hp(),
    )
    b = adapter_key(
        train_sha1="x", valid_sha1="y",
        base_model="meta-llama/Meta-Llama-3.2-3B-Instruct",
        hyperparams=_hp(),
    )
    assert a != b


def test_dataset_prefix_is_stable_and_distinct():
    p1 = dataset_prefix("a", "b")
    p2 = dataset_prefix("a", "b")
    p3 = dataset_prefix("b", "a")
    assert p1 == p2
    assert p1 != p3
    assert p1.startswith("datasets/")


def test_adapter_prefix_format():
    assert adapter_prefix("deadbeef") == "adapters/deadbeef"
