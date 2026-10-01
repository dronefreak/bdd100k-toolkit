"""Latency measurement for the demos."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from bdd100k_toolkit.classification.predict import Prediction, Predictor


def median_prediction(
    predictor: Predictor, image: Image.Image, runs: int
) -> Prediction:
    """
    Predict ``runs`` times after a warm-up and return the median-latency result.

    One timed call is noisy (it can be 50 times slower than steady state).
    """
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    predictor.predict(image)
    results = [predictor.predict(image) for _ in range(runs)]
    middle = statistics.median_low(r.seconds for r in results)
    return next(r for r in results if r.seconds == middle)
