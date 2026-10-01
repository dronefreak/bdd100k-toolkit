"""Tests for ``bdd100k_toolkit.utils.split``."""

from __future__ import annotations

from bdd100k_toolkit.utils.split import seeded_holdout


def test_seeded_holdout_splits_all_keys() -> None:
    keys = [f"k{i}" for i in range(20)]
    train_keys, val_keys = seeded_holdout(keys, 0.15, seed=42)
    assert sorted(train_keys + val_keys) == sorted(keys)
    assert set(train_keys).isdisjoint(val_keys)


def test_seeded_holdout_is_deterministic() -> None:
    keys = [f"k{i}" for i in range(20)]
    result_a = seeded_holdout(keys, 0.15, seed=42)
    result_b = seeded_holdout(keys, 0.15, seed=42)
    assert result_a == result_b


def test_seeded_holdout_always_holds_out_at_least_one() -> None:
    keys = ["a", "b", "c"]
    _, val_keys = seeded_holdout(keys, 0.01, seed=1)
    assert len(val_keys) >= 1


def test_seeded_holdout_empty_input() -> None:
    train_keys, val_keys = seeded_holdout([], 0.15, seed=42)
    assert train_keys == []
    assert val_keys == []


def test_seeded_holdout_preserves_original_order() -> None:
    keys = ["a", "b", "c", "d", "e", "f", "g"]
    train_keys, val_keys = seeded_holdout(keys, 0.3, seed=7)
    assert train_keys == [k for k in keys if k in train_keys]
    assert val_keys == [k for k in keys if k in val_keys]
