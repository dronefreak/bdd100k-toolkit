"""
BDD100K scenario (scene) classification (unofficial task).

A 7-class driving-scenario task (city street / highway / residential / parking
lot / gas stations / tunnel / unknown) built from BDD100K's per-image
``attributes.scene`` field. The task follows the ``bdd100k-scenario-
classification`` Kaggle dataset (marquis03); see ``_bdd100k_common.py`` for the
two accepted input layouts, the ``unknown`` class (the official ``undefined``
value, kept by default) and the splits.
"""

from __future__ import annotations

from pathlib import Path

from bdd100k_toolkit.classification.base import (
    ClassificationAdapter,
    ClassificationSpec,
)
from bdd100k_toolkit.classification.datasets._bdd100k_common import (
    UNKNOWN_CLASS,
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
        classes=[*_CLASSES, UNKNOWN_CLASS],
        description=(
            "7-class driving-scenario classification (city street / highway / "
            "residential / parking lot / gas stations / tunnel / unknown) derived "
            "from BDD100K's per-image attributes.scene field. Unofficial task; "
            "follows the Kaggle dataset of the same name."
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

    def prepare_classification(
        self,
        raw_dir: Path,
        output_dir: Path,
        *,
        include_unknown: bool = True,
        labels_dir: Path | None = None,
    ) -> None:
        """Convert a BDD100K download into canonical scenario splits."""
        prepare_attribute_classification(
            raw_dir,
            output_dir,
            "scene",
            _CLASSES,
            include_unknown=include_unknown,
            labels_dir=labels_dir,
        )
