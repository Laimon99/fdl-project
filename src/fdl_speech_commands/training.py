from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from .config import ExperimentConfig, load_config
from .constants import PROJECT_ROOT
from .datasets import build_dataset, read_manifest
from .features import feature_shape
from .models import build_model, write_model_summary
from .utils import (
    ProjectError,
    ensure_directory,
    runtime_metadata,
    set_global_determinism,
    write_json,
)

console = Console()


class ValidationMetrics(keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        labels = np.concatenate([batch_labels.numpy() for _, batch_labels in self.dataset])
        logits = self.model.predict(self.dataset, verbose=0)
        predictions = np.argmax(logits, axis=1)
        logs["val_macro_f1"] = f1_score(labels, predictions, average="macro", zero_division=0)
        logs["val_posthoc_accuracy"] = accuracy_score(labels, predictions)
        console.print(
            f" - val_macro_f1: {logs['val_macro_f1']:.4f}"
            f" - val_posthoc_accuracy: {logs['val_posthoc_accuracy']:.4f}"
        )


def _compile_model(model: keras.Model, config: ExperimentConfig) -> None:
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        global_clipnorm=5.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def train_experiment(
    config_path: str | Path,
    overwrite: bool = False,
    epochs_override: int | None = None,
) -> Path:
    config = load_config(config_path)
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / config.experiment_id
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise ProjectError(
                f"Run already exists: {run_dir}. Use --overwrite to replace it deliberately."
            )
        shutil.rmtree(run_dir)
    ensure_directory(run_dir)
    set_global_determinism(config.seed)

    manifest = read_manifest(config.manifest)
    augmentation = config.augmentation if config.training.augment else None
    train_dataset = build_dataset(
        manifest,
        config.raw_dir,
        config.features,
        augmentation_config=augmentation,
        split="training",
        batch_size=config.training.batch_size,
        training=True,
        seed=config.seed,
    )
    validation_dataset = build_dataset(
        manifest,
        config.raw_dir,
        config.features,
        split="validation",
        batch_size=config.training.batch_size,
        training=False,
        seed=config.seed,
    )
    normalization_dataset = build_dataset(
        manifest,
        config.raw_dir,
        config.features,
        split="training",
        batch_size=config.training.batch_size,
        training=False,
        seed=config.seed,
    )

    model, normalizer = build_model(feature_shape(config.features), config.model)
    console.print("Adapting normalization statistics on the unaugmented training split...")
    normalizer.adapt(normalization_dataset.map(lambda features, _: features))
    _compile_model(model, config)
    write_model_summary(model, run_dir / "model_summary.txt")
    write_json(run_dir / "environment.json", runtime_metadata())
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.as_dict(), stream, sort_keys=False)

    checkpoint = run_dir / "best_model.keras"
    callbacks: list[keras.callbacks.Callback] = [
        ValidationMetrics(validation_dataset),
        keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=config.training.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, config.training.patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
        keras.callbacks.TensorBoard(
            log_dir=run_dir / "tensorboard",
            histogram_freq=0,
            update_freq="epoch",
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    epochs = epochs_override or config.training.epochs
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index_label="epoch")

    if not checkpoint.exists():
        model.save(checkpoint)
    best_model = keras.models.load_model(checkpoint)
    labels = np.concatenate([labels.numpy() for _, labels in validation_dataset])
    logits = best_model.predict(validation_dataset, verbose=0)
    predictions = np.argmax(logits, axis=1)
    validation_metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "epochs_completed": len(history.epoch),
        "parameters": best_model.count_params(),
    }
    write_json(run_dir / "validation_metrics.json", validation_metrics)
    console.print(f"[green]Completed experiment:[/] {config.experiment_id}")
    return run_dir


def train_all(
    config_dir: str | Path = PROJECT_ROOT / "configs",
    overwrite: bool = False,
    epochs_override: int | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    for config_path in sorted(Path(config_dir).glob("e*.yaml")):
        outputs.append(train_experiment(config_path, overwrite, epochs_override))
    if not outputs:
        raise ProjectError(f"No experiment configs found below {config_dir}")
    return outputs

