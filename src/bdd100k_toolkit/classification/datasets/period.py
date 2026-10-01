"""
BDD100K period (time-of-day) classification (unofficial task).

A 4-class time-of-day task (daytime / night / dawn or dusk / unknown) built
from BDD100K's per-image ``attributes.timeofday`` field. The task follows the
``bdd100k-period-classification`` Kaggle dataset (marquis03); see
``_bdd100k_common.py`` for the two accepted input layouts, the ``unknown``
class (the official ``undefined`` value, kept by default) and the splits.
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
from bdd100k_toolkit.utils.io import ImageOptions

# "dawn/dusk" is BDD100K's raw value; the class (folder) name avoids the slash.
_CLASSES = ["daytime", "night", "dawn or dusk"]
_ALIASES = {"dawn/dusk": "dawn or dusk"}


@register
class BDD100KPeriodAdapter(ClassificationAdapter):
    """Adapter for BDD100K time-of-day (period) classification."""

    spec = ClassificationSpec(
        key="bdd100k-period",
        display_name="BDD100K Period (Time-of-Day) Classification",
        classes=[*_CLASSES, UNKNOWN_CLASS],
        description=(
            "4-class time-of-day classification (daytime / night / dawn or dusk / "
            "unknown) derived from BDD100K's per-image attributes.timeofday "
            "field. Unofficial task; follows the Kaggle dataset of the same name."
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
        image_options: ImageOptions | None = None,
    ) -> None:
        """Convert a BDD100K download into canonical period splits."""
        prepare_attribute_classification(
            raw_dir,
            output_dir,
            "timeofday",
            _CLASSES,
            aliases=_ALIASES,
            include_unknown=include_unknown,
            labels_dir=labels_dir,
            image_options=image_options,
        )
