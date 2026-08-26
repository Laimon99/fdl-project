from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import ModelConfig
from .constants import LABELS


def _conv_bn_relu(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    name: str = "conv",
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _separable_block(
    inputs: tf.Tensor,
    filters: int,
    strides: tuple[int, int],
    name: str,
) -> tf.Tensor:
    x = layers.DepthwiseConv2D(
        (3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name}_depthwise",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_depthwise_bn")(x)
    x = layers.ReLU(name=f"{name}_depthwise_relu")(x)
    x = layers.Conv2D(
        filters,
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_pointwise",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pointwise_bn")(x)
    return layers.ReLU(name=f"{name}_pointwise_relu")(x)


def _build_mlp(inputs: tf.Tensor, dropout: float) -> tf.Tensor:
    x = layers.Flatten(name="flatten_features")(inputs)
    x = layers.Dense(256, use_bias=False, kernel_initializer="he_normal", name="dense_1")(x)
    x = layers.BatchNormalization(name="dense_1_bn")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(dropout, name="dense_1_dropout")(x)
    x = layers.Dense(128, use_bias=False, kernel_initializer="he_normal", name="dense_2")(x)
    x = layers.BatchNormalization(name="dense_2_bn")(x)
    x = layers.ReLU(name="dense_2_relu")(x)
    return layers.Dropout(dropout, name="dense_2_dropout")(x)


def _build_small_cnn(inputs: tf.Tensor, dropout: float) -> tf.Tensor:
    x = _conv_bn_relu(inputs, 32, (5, 5), name="stem")
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    x = _conv_bn_relu(x, 64, (3, 3), name="conv_2")
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    x = _conv_bn_relu(x, 96, (3, 3), name="conv_3")
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    return layers.Dropout(dropout, name="head_dropout")(x)


def _build_ds_cnn(inputs: tf.Tensor, dropout: float) -> tf.Tensor:
    x = _conv_bn_relu(inputs, 48, (5, 5), strides=(2, 2), name="stem")
    x = _separable_block(x, 64, (1, 1), "ds_1")
    x = _separable_block(x, 96, (2, 2), "ds_2")
    x = _separable_block(x, 128, (1, 1), "ds_3")
    # Equal spatial strides keep TensorFlow's depthwise CPU kernel portable across
    # Windows and Linux (some backends reject asymmetric depthwise strides).
    x = _separable_block(x, 128, (2, 2), "ds_4")
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    return layers.Dropout(dropout, name="head_dropout")(x)


def _build_crnn(inputs: tf.Tensor, dropout: float) -> tf.Tensor:
    x = _conv_bn_relu(inputs, 32, (5, 5), name="stem")
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    x = _conv_bn_relu(x, 64, (3, 3), name="conv_2")
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    time_steps, frequency_bins, channels = x.shape[1], x.shape[2], x.shape[3]
    if None in (time_steps, frequency_bins, channels):
        raise ValueError("CRNN requires a statically known feature shape")
    x = layers.Reshape((time_steps, frequency_bins * channels), name="frequency_flatten")(x)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=dropout / 2),
        merge_mode="concat",
        name="bidirectional_gru",
    )(x)
    x = layers.GlobalAveragePooling1D(name="temporal_average_pool")(x)
    return layers.Dropout(dropout, name="head_dropout")(x)


def build_model(
    input_shape: tuple[int, int, int],
    config: ModelConfig,
    num_classes: int = len(LABELS),
) -> tuple[keras.Model, layers.Normalization]:
    inputs = keras.Input(shape=input_shape, name="audio_features")
    normalization = layers.Normalization(axis=None, name="training_set_normalization")
    x = normalization(inputs)
    builders = {
        "mlp": _build_mlp,
        "small_cnn": _build_small_cnn,
        "ds_cnn": _build_ds_cnn,
        "crnn": _build_crnn,
    }
    representation = builders[config.name](x, config.dropout)
    logits = layers.Dense(num_classes, dtype="float32", name="command_logits")(representation)
    model = keras.Model(inputs, logits, name=f"speech_commands_{config.name}")
    return model, normalization


def write_model_summary(model: keras.Model, path: str | Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True, show_trainable=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
