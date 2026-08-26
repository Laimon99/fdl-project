from __future__ import annotations

import tensorflow as tf

from .config import FeatureConfig


def feature_shape(config: FeatureConfig) -> tuple[int, int, int]:
    frames = 1 + (config.clip_samples - config.frame_length) // config.frame_step
    bins = config.mfcc_bins if config.kind == "mfcc" else config.mel_bins
    return frames, bins, 1


def log_mel_spectrogram(waveform: tf.Tensor, config: FeatureConfig) -> tf.Tensor:
    waveform = tf.ensure_shape(tf.cast(waveform, tf.float32), [config.clip_samples])
    stft = tf.signal.stft(
        waveform,
        frame_length=config.frame_length,
        frame_step=config.frame_step,
        fft_length=config.fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power_spectrogram = tf.square(tf.abs(stft))
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.mel_bins,
        num_spectrogram_bins=config.fft_length // 2 + 1,
        sample_rate=config.sample_rate,
        lower_edge_hertz=config.lower_hertz,
        upper_edge_hertz=config.upper_hertz,
        dtype=tf.float32,
    )
    mel = tf.matmul(power_spectrogram, mel_matrix)
    log_mel = tf.math.log(mel + tf.keras.backend.epsilon())
    frames = 1 + (config.clip_samples - config.frame_length) // config.frame_step
    return tf.ensure_shape(log_mel, [frames, config.mel_bins])


def extract_features(waveform: tf.Tensor, config: FeatureConfig) -> tf.Tensor:
    log_mel = log_mel_spectrogram(waveform, config)
    if config.kind == "mfcc":
        features = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., : config.mfcc_bins]
    elif config.kind == "log_mel":
        features = log_mel
    else:
        raise ValueError(f"Unsupported feature kind: {config.kind}")
    features = features[..., tf.newaxis]
    return tf.ensure_shape(features, feature_shape(config))
