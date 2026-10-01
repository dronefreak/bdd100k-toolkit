"""Tests for ``bdd100k_toolkit.utils.bitmask``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bdd100k_toolkit.utils.bitmask import decode_bitmask


def _write_bitmask(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def test_decode_bitmask_single_instance(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 0] = 5  # category id 5
    arr[1:3, 1:3, 2] = 0  # B
    arr[1:3, 1:3, 3] = 1  # A -> instance_id = (0<<8)+1 = 1
    path = tmp_path / "mask.png"
    _write_bitmask(path, arr)

    instances = decode_bitmask(path)
    assert len(instances) == 1
    inst = instances[0]
    assert inst.category_id == 5
    assert inst.instance_id == 1
    assert inst.mask.sum() == 4
    assert not inst.is_crowd
    assert not inst.is_ignored


def test_decode_bitmask_background_ignored(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)  # category 0 everywhere = background
    path = tmp_path / "empty.png"
    _write_bitmask(path, arr)
    assert decode_bitmask(path) == []


def test_decode_bitmask_crowd_and_ignored_bits(tmp_path: Path) -> None:
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[..., 0] = 1  # category
    arr[..., 1] = 0b0011  # crowd bit (bit 1) + ignored bit (bit 0) both set
    arr[..., 3] = 7  # instance id
    path = tmp_path / "crowd.png"
    _write_bitmask(path, arr)

    instances = decode_bitmask(path)
    assert len(instances) == 1
    assert instances[0].is_crowd
    assert instances[0].is_ignored


def test_decode_bitmask_multiple_instances_same_category(tmp_path: Path) -> None:
    arr = np.zeros((2, 4, 4), dtype=np.uint8)
    arr[:, 0:2, 0] = 2  # category 2 for first instance region
    arr[:, 0:2, 3] = 1  # instance id 1
    arr[:, 2:4, 0] = 2  # category 2 for second instance region
    arr[:, 2:4, 3] = 2  # instance id 2
    path = tmp_path / "multi.png"
    _write_bitmask(path, arr)

    instances = decode_bitmask(path)
    assert len(instances) == 2
    ids = sorted(inst.instance_id for inst in instances)
    assert ids == [1, 2]
