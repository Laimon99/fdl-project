import numpy as np
import pytest
import tensorflow as tf

from fdl_speech_commands.augmentation import augment_waveform, spec_augment
from fdl_speech_commands.config import AugmentationConfig, FeatureConfig, ModelConfig
from fdl_speech_commands.constants import LABELS
from fdl_speech_commands.features import extract_features, feature_shape
from fdl_speech_commands.models import build_model


@pytest.mark.parametrize(
    ("kind", "expected_shape"),
    [("log_mel", (98, 40, 1)), ("mfcc", (98, 13, 1))],
)
def test_feature_shape_and_finiteness(kind: str, expected_shape: tuple[int, ...]) -> None:
    config = FeatureConfig(kind=kind)
    time = tf.linspace(0.0, 1.0, 16_000)
    waveform = tf.sin(2 * np.pi * 440.0 * time)
    features = extract_features(waveform, config)
    assert tuple(features.shape) == expected_shape
    assert np.isfinite(features.numpy()).all()


@pytest.mark.parametrize(
    ("kind", "name"),
    [("mfcc", "mlp"), ("log_mel", "small_cnn"), ("log_mel", "ds_cnn"), ("log_mel", "crnn")],
)
def test_model_forward_path(kind: str, name: str) -> None:
    config = FeatureConfig(kind=kind)
    model, normalizer = build_model(feature_shape(config), ModelConfig(name=name))
    batch = tf.zeros((2, *feature_shape(config)), dtype=tf.float32)
    normalizer.adapt(batch)
    logits = model(batch, training=False)
    assert tuple(logits.shape) == (2, len(LABELS))
    assert model.count_params() > 0


def test_augmentations_preserve_shapes() -> None:
    feature_config = FeatureConfig()
    augmentation = AugmentationConfig()
    waveform = tf.linspace(-0.2, 0.2, feature_config.clip_samples)
    noise_bank = tf.random.normal((4, feature_config.clip_samples), stddev=0.02)
    augmented = augment_waveform(waveform, noise_bank, feature_config, augmentation)
    features = extract_features(augmented, feature_config)
    masked = spec_augment(features, augmentation)
    assert tuple(augmented.shape) == (feature_config.clip_samples,)
    assert tuple(masked.shape) == feature_shape(feature_config)
    assert np.isfinite(masked.numpy()).all()

