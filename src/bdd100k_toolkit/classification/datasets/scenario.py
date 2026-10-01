"""
BDD100K scenario (scene) classification.

Derives a 6-class driving-scenario classification task from BDD100K's native
per-image ``attributes.scene`` field; see
``bdd100k-scenario-classification`` on Kaggle (marquis03) for the third-party
re-export this mirrors; this adapter derives the same task directly from the
official label release instead (see ``_bdd100k_common.py`` docstring for why).

``undefined`` scene values are dropped (not a decidable class).
"""

from __future__ import annotations

from pathlib import Path

from bdd100k_toolkit.classification.base import (
    ClassificationAdapter,
    ClassificationSpec,
)
from bdd100k_toolkit.classification.datasets._bdd100k_common import (
    prepare_attribute_classification,
)
from bdd100k_toolkit.classification.registry import register

_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]


@register
class BDD100KScenarioAdapter(ClassificationAdapter):
    """Adapter for BDD100K driving-scenario (scene) classification."""

    spec = ClassificationSpec(
        key="bdd100k-scenario",
        display_name="BDD100K Scenario Classification",
        classes=_CLASSES,
        description=(
            "6-class driving-scenario classification (city street / "
            "highway / residential / parking lot / gas stations / tunnel) "
            "derived from BDD100K's per-image attributes.scene field."
        ),
        homepage="https://www.bdd100k.com/",
        citation=(
            "@inproceedings{yu2020bdd100k,\n"
            "  title={BDD100K: A Diverse Driving Dataset for Heterogeneous "
            "Multitask Learning},\n"
            "  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, "
            "Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, "
            "Vashisht and Darrell, Trevor},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer "
            "Vision and Pattern Recognition},\n"
            "  pages={2636--2645},\n"
            "  year={2020}\n"
            "}"
        ),
        license=(
            "BDD100K License (non-commercial research/education; "
            "registration-gated, no redistribution)."
        ),
    )

    def prepare_classification(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official BDD100K release into canonical scenario splits."""
        prepare_attribute_classification(raw_dir, output_dir, "scene", _CLASSES)
