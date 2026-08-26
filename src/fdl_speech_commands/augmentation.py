from __future__ import annotations

import tensorflow as tf

from .config import AugmentationConfig, FeatureConfig


def _time_shift(waveform: tf.Tensor, max_shift_samples: int) -> tf.Tensor:
    if max_shift_samples <= 0:
        return waveform
    shift = tf.random.uniform([], -max_shift_samples, max_shift_samples + 1, dtype=tf.int32)

    def shift_right() -> tf.Tensor:
        return tf.concat([tf.zeros([shift], tf.float32), waveform[:-shift]], axis=0)

    def shift_left() -> tf.Tensor:
        amount = -shift
        return tf.concat([waveform[amount:], tf.zeros([amount], tf.float32)], axis=0)

    return tf.case(
        [(shift > 0, shift_right), (shift < 0, shift_left)],
        default=lambda: waveform,
        exclusive=True,
    )


def _mix_at_snr(
    waveform: tf.Tensor,
    noise: tf.Tensor,
    snr_db_min: float,
    snr_db_max: float,
) -> tf.Tensor:
    epsilon = tf.constant(1e-7, tf.float32)
    signal_rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)) + epsilon)
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise)) + epsilon)
    snr_db = tf.random.uniform([], snr_db_min, snr_db_max)
    reference_rms = tf.maximum(signal_rms, tf.constant(0.025, tf.float32))
    scale = reference_rms / (noise_rms * tf.pow(10.0, snr_db / 20.0))
    return tf.clip_by_value(waveform + noise * scale, -1.0, 1.0)


def augment_waveform(
    waveform: tf.Tensor,
    noise_bank: tf.Tensor | None,
    feature_config: FeatureConfig,
    config: AugmentationConfig,
) -> tf.Tensor:
    waveform = tf.ensure_shape(tf.cast(waveform, tf.float32), [feature_config.clip_samples])
    max_shift = round(config.max_shift_ms * feature_config.sample_rate / 1000)
    waveform = _time_shift(waveform, max_shift)
    gain = tf.random.uniform([], config.gain_min, config.gain_max)
    waveform = tf.clip_by_value(waveform * gain, -1.0, 1.0)

    if noise_bank is not None:
        apply_noise = tf.random.uniform([]) < config.noise_probability

        def noisy() -> tf.Tensor:
            index = tf.random.uniform([], 0, tf.shape(noise_bank)[0], dtype=tf.int32)
            return _mix_at_snr(
                waveform,
                noise_bank[index],
                config.snr_db_min,
                config.snr_db_max,
            )

        waveform = tf.cond(apply_noise, noisy, lambda: waveform)
    return tf.ensure_shape(waveform, [feature_config.clip_samples])


def spec_augment(features: tf.Tensor, config: AugmentationConfig) -> tf.Tensor:
    features = tf.identity(features)
    time_steps = tf.shape(features)[0]
    frequency_bins = tf.shape(features)[1]

    for _ in range(config.time_masks):
        width = tf.random.uniform([], 0, config.time_mask_max + 1, dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(1, time_steps - width + 1), dtype=tf.int32)
        mask = tf.logical_and(tf.range(time_steps) >= start, tf.range(time_steps) < start + width)
        features = tf.where(mask[:, tf.newaxis, tf.newaxis], 0.0, features)

    for _ in range(config.frequency_masks):
        width = tf.random.uniform([], 0, config.frequency_mask_max + 1, dtype=tf.int32)
        start = tf.random.uniform(
            [], 0, tf.maximum(1, frequency_bins - width + 1), dtype=tf.int32
        )
        mask = tf.logical_and(
            tf.range(frequency_bins) >= start,
            tf.range(frequency_bins) < start + width,
        )
        features = tf.where(mask[tf.newaxis, :, tf.newaxis], 0.0, features)
    return features
