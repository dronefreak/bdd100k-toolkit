"""
Shared seeded train/valid split helper.

Every BDD100K-Toolkit adapter that only gets an official ``train``/``val``
split (BDD100K never releases a labelled ``test`` split) follows the same
convention: keep official ``val`` as canonical ``test``, and carve a seeded
validation slice out of official ``train``. This was previously duplicated
per-adapter (classification, detection); factored out here once a third
adapter (semantic segmentation) needed the identical logic.
"""

from __future__ import annotations

import random


def seeded_holdout(
    keys: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """
    Split ``keys`` into ``(train_keys, val_keys)`` using a seeded shuffle.

    Args:
        keys: Sorted, deterministic list of item identifiers (e.g. file stems).
        val_fraction: Fraction of ``keys`` to hold out for validation.
        seed: Seed for the deterministic shuffle.

    Returns:
        A ``(train_keys, val_keys)`` tuple; ``val_keys`` has at least 1 entry
        whenever ``keys`` is non-empty.

    """
    shuffled = keys[:]
    random.Random(seed).shuffle(shuffled)  # noqa: S311  # nosec: B311
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val_keys = set(shuffled[:n_val])
    train_keys = [k for k in keys if k not in val_keys]
    val_keys_ordered = [k for k in keys if k in val_keys]
    return train_keys, val_keys_ordered
