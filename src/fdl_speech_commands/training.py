from __future__ import annotations

import shutil
from dataclasses import replace
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
    experiment_id_override: str | None = None,
    seed_override: int | None = None,
) -> Path:
    config = load_config(config_path)
    config = replace(
        config,
        experiment_id=experiment_id_override or config.experiment_id,
        seed=config.seed if seed_override is None else seed_override,
    )
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
    normalization_dataset = train_dataset
    if augmentation is not None:
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


def repeat_selected_seeds(
    overwrite: bool = False,
    seeds: tuple[int, ...] = (7, 21, 42),
) -> pd.DataFrame:
    """Repeat the validation-selected configuration to quantify seed sensitivity."""
    from .evaluation import build_leaderboard

    leaderboard = build_leaderboard()
    best_id = str(leaderboard.iloc[0]["experiment_id"])
    source_config = PROJECT_ROOT / "configs" / f"{best_id}.yaml"
    if not source_config.exists():
        raise ProjectError(f"Cannot locate source config for selected run: {best_id}")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = best_id if seed == 42 else f"{best_id}_seed{seed}"
        run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
        if not (run_dir / "validation_metrics.json").exists():
            train_experiment(
                source_config,
                overwrite=overwrite,
                experiment_id_override=run_id,
                seed_override=seed,
            )
        import json

        with (run_dir / "validation_metrics.json").open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        rows.append({"selected_config": best_id, "run_id": run_id, "seed": seed, **metrics})

    frame = pd.DataFrame(rows).sort_values("seed")
    output = PROJECT_ROOT / "artifacts" / "tables" / "seed_stability_validation.csv"
    ensure_directory(output.parent)
    frame.to_csv(output, index=False)
    summary = {
        "selected_config": best_id,
        "seeds": list(seeds),
        "validation_accuracy_mean": frame["accuracy"].mean(),
        "validation_accuracy_std": frame["accuracy"].std(ddof=1),
        "validation_macro_f1_mean": frame["macro_f1"].mean(),
        "validation_macro_f1_std": frame["macro_f1"].std(ddof=1),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return frame
