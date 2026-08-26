import numpy as np

from fdl_speech_commands.evaluation import (
    expected_calibration_error,
    stratified_bootstrap_intervals,
)


def test_perfect_predictions_have_zero_ece() -> None:
    labels = np.arange(12)
    probabilities = np.eye(12)
    ece, bins = expected_calibration_error(labels, probabilities)
    assert ece == 0.0
    assert bins["count"].sum() == len(labels)


def test_bootstrap_intervals_cover_perfect_point_estimate() -> None:
    labels = np.repeat(np.arange(12), 3)
    predictions = labels.copy()
    intervals = stratified_bootstrap_intervals(labels, predictions, resamples=100, seed=1)
    assert intervals["accuracy"]["point"] == 1.0
    assert intervals["accuracy"]["lower_95"] == 1.0
    assert intervals["macro_f1"]["upper_95"] == 1.0

