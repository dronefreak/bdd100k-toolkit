"""
BDD100K weather classification (unofficial task).

A 7-class weather task (clear / partly cloudy / overcast / rainy / snowy /
foggy / unknown) built from BDD100K's per-image ``attributes.weather`` field.
The task follows the ``bdd100k-weather-classification`` Kaggle dataset
(marquis03); see ``_bdd100k_common.py`` for the two accepted input layouts, the
``unknown`` class (the official ``undefined`` value, kept by default) and the
splits.
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

_CLASSES = ["clear", "partly cloudy", "overcast", "rainy", "snowy", "foggy"]


@register
class BDD100KWeatherAdapter(ClassificationAdapter):
    """Adapter for BDD100K weather classification."""

    spec = ClassificationSpec(
        key="bdd100k-weather",
        display_name="BDD100K Weather Classification",
        classes=[*_CLASSES, UNKNOWN_CLASS],
        description=(
            "7-class weather classification (clear / partly cloudy / overcast / "
            "rainy / snowy / foggy / unknown) derived from BDD100K's per-image "
            "attributes.weather field. Unofficial task; follows the Kaggle "
            "dataset of the same name."
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
            "BDD100K License (UC Regents): educational, research and "
            "not-for-profit use and redistribution with the copyright notice "
            "carried forward; commercial use needs separate permission."
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
        """Convert a BDD100K download into canonical weather splits."""
        prepare_attribute_classification(
            raw_dir,
            output_dir,
            "weather",
            _CLASSES,
            include_unknown=include_unknown,
            labels_dir=labels_dir,
            image_options=image_options,
        )
