from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import tensorflow as tf
from matplotlib import pyplot as plt
from rich.console import Console
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from tensorflow import keras

from .config import ExperimentConfig, load_config
from .constants import CLIP_SAMPLES, LABELS, PROJECT_ROOT, SAMPLE_RATE
from .datasets import build_dataset, read_manifest
from .features import extract_features
from .utils import ProjectError, ensure_directory, write_json

console = Console()


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> tuple[float, pd.DataFrame]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    records: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        else:
            accuracy = mean_confidence = float("nan")
        records.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return float(ece), pd.DataFrame(records)


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == index) for index in range(len(LABELS))]
    accuracy_values = np.empty(resamples, dtype=np.float64)
    f1_values = np.empty(resamples, dtype=np.float64)
    for resample in range(resamples):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        sampled_labels = labels[indices]
        sampled_predictions = predictions[indices]
        accuracy_values[resample] = np.mean(sampled_labels == sampled_predictions)
        f1_values[resample] = f1_score(
            sampled_labels,
            sampled_predictions,
            average="macro",
            labels=np.arange(len(LABELS)),
            zero_division=0,
        )

    def summarize(values: np.ndarray, point: float) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {"point": point, "lower_95": float(lower), "upper_95": float(upper)}

    return {
        "accuracy": summarize(accuracy_values, accuracy_score(labels, predictions)),
        "macro_f1": summarize(
            f1_values, f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "resamples": resamples,
        "seed": seed,
    }


def _plot_confusion(labels: np.ndarray, predictions: np.ndarray, output: Path) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(LABELS)))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for axis, values, title, fmt in (
        (axes[0], matrix, "Test confusion matrix - counts", "d"),
        (axes[1], normalized, "Test confusion matrix - row normalized", ".2f"),
    ):
        sns.heatmap(
            values,
            annot=True,
            fmt=fmt,
            cmap="mako",
            xticklabels=LABELS,
            yticklabels=LABELS,
            square=True,
            cbar=False,
            ax=axis,
            annot_kws={"size": 8},
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_reliability(calibration: pd.DataFrame, output: Path) -> None:
    valid = calibration[calibration["count"] > 0]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
    axis.plot(
        valid["mean_confidence"],
        valid["accuracy"],
        marker="o",
        linewidth=2,
        color="#2563eb",
        label="Model",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy")
    axis.set_title("Reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_clip(row: pd.Series, raw_dir: Path) -> np.ndarray:
    audio, sample_rate = sf.read(raw_dir / row["path"], dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    offset = int(row["offset_samples"]) if row["source_type"] == "background_slice" else 0
    clip = audio[offset : offset + CLIP_SAMPLES]
    return np.pad(clip, (0, max(0, CLIP_SAMPLES - len(clip))))[:CLIP_SAMPLES]


def _qualitative_gallery(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    merged = predictions.merge(manifest, on="example_id", validate="one_to_one")
    mistakes = merged[~merged["correct"]].nlargest(6, "confidence")
    uncertain_correct = merged[merged["correct"]].nsmallest(6, "confidence")
    selected = pd.concat(
        [mistakes.assign(selection="high_confidence_error"), uncertain_correct.assign(selection="uncertain_correct")],
        ignore_index=True,
    )
    audio_dir = ensure_directory(output_dir / "qualitative_audio")
    figure, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for axis, (_, row) in zip(axes.flat, selected.iterrows(), strict=False):
        clip = _load_clip(row, raw_dir)
        mel = librosa.feature.melspectrogram(
            y=clip,
            sr=SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            win_length=480,
            n_mels=40,
            fmin=20,
            fmax=7_600,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        axis.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(
            f"true: {row['true_label']} | pred: {row['predicted_label']}\n"
            f"confidence {row['confidence']:.2f} - {row['selection'].replace('_', ' ')}",
            fontsize=9,
        )
        axis.set_xlabel("Time frame")
        axis.set_ylabel("Mel bin")
        filename = f"{row['selection']}_{row['example_id']}.wav"
        sf.write(audio_dir / filename, clip, SAMPLE_RATE, subtype="PCM_16")
        records.append(
            {
                "selection": row["selection"],
                "example_id": row["example_id"],
                "source_path": row["path"],
                "audio_clip": f"qualitative_audio/{filename}",
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }
        )
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Qualitative audit: confident failures and ambiguous successes", fontsize=16)
    figure.savefig(output_dir / "qualitative_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(records).to_csv(output_dir / "qualitative_samples.csv", index=False)


def _measure_latency(
    model: keras.Model,
    dataset: tf.data.Dataset,
    config: ExperimentConfig,
    waveform: np.ndarray,
    repeats: int = 200,
) -> dict[str, float]:
    features, _ = next(iter(dataset.unbatch().batch(1)))

    @tf.function
    def model_only(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    @tf.function
    def end_to_end(single_waveform: tf.Tensor) -> tf.Tensor:
        extracted = extract_features(single_waveform, config.features)[tf.newaxis]
        return model(extracted, training=False)

    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

    def time_call(function, argument) -> np.ndarray:
        for _ in range(20):
            _ = function(argument).numpy()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = function(argument).numpy()
            timings.append((time.perf_counter() - start) * 1_000)
        return np.asarray(timings)

    model_values = time_call(model_only, features)
    end_to_end_values = time_call(end_to_end, waveform_tensor)
    return {
        "batch_size": 1,
        "repeats": repeats,
        "median_ms": float(np.median(end_to_end_values)),
        "p95_ms": float(np.quantile(end_to_end_values, 0.95)),
        "mean_ms": float(np.mean(end_to_end_values)),
        "model_only_median_ms": float(np.median(model_values)),
        "model_only_p95_ms": float(np.quantile(model_values, 0.95)),
        "scope": "compiled feature extraction plus neural model on CPU",
    }


def _robustness_sweep(
    model: keras.Model,
    config: ExperimentConfig,
    manifest: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    conditions = [
        ("clean", None, 0),
        ("background_20db", 20.0, 0),
        ("background_10db", 10.0, 0),
        ("background_0db", 0.0, 0),
        ("shift_plus_100ms", None, 1_600),
        ("shift_minus_100ms", None, -1_600),
    ]
    for name, snr, shift in conditions:
        dataset = build_dataset(
            manifest,
            config.raw_dir,
            config.features,
            split=split,
            batch_size=config.training.batch_size,
            corruption_snr_db=snr,
            time_shift_samples=shift,
        )
        labels = np.concatenate([batch.numpy() for _, batch in dataset])
        predictions = np.argmax(model.predict(dataset, verbose=0), axis=1)
        records.append(
            {
                "condition": name,
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "robustness.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_frame = frame.melt("condition", value_vars=["accuracy", "macro_f1"], var_name="metric")
    sns.barplot(data=plot_frame, x="condition", y="value", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_title("Robustness under deterministic audio corruptions")
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(output_dir / "robustness.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return frame


def evaluate_run(
    run_dir: str | Path,
    split: str = "testing",
    bootstrap_resamples: int = 10_000,
    robustness: bool = True,
) -> Path:
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "config_resolved.yaml")
    model_path = run_dir / "best_model.keras"
    if not model_path.exists():
        raise ProjectError(f"Missing trained model: {model_path}")
    model = keras.models.load_model(model_path)
    manifest = read_manifest(config.manifest)
    subset = manifest[manifest["split"] == split].reset_index(drop=True)
    dataset = build_dataset(
        manifest,
        config.raw_dir,
        config.features,
        split=split,
        batch_size=config.training.batch_size,
    )
    labels = np.concatenate([batch.numpy() for _, batch in dataset])
    logits = model.predict(dataset, verbose=1)
    probabilities = _probabilities(logits)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    output_dir = ensure_directory(run_dir / f"evaluation_{split}")
    predictions = pd.DataFrame(
        {
            "example_id": subset["example_id"],
            "true_index": labels,
            "true_label": [LABELS[index] for index in labels],
            "predicted_index": predicted,
            "predicted_label": [LABELS[index] for index in predicted],
            "confidence": confidence,
            "correct": labels == predicted,
        }
    )
    for index, label in enumerate(LABELS):
        predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    report = classification_report(
        labels,
        predicted,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    ece, calibration = expected_calibration_error(labels, probabilities)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
    intervals = stratified_bootstrap_intervals(
        labels, predicted, resamples=bootstrap_resamples
    )
    metrics = {
        "split": split,
        "examples": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "macro_precision": precision_score(labels, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predicted, average="weighted", zero_division=0),
        "negative_log_likelihood": log_loss(labels, probabilities, labels=np.arange(len(LABELS))),
        "expected_calibration_error_15_bins": ece,
        "bootstrap_confidence_intervals": intervals,
        "parameters": model.count_params(),
        "model_size_bytes": model_path.stat().st_size,
        "latency": _measure_latency(
            model,
            dataset,
            config,
            _load_clip(subset.iloc[0], config.raw_dir),
        ),
    }
    write_json(output_dir / "metrics.json", metrics)
    _plot_confusion(labels, predicted, output_dir / "confusion_matrices.png")
    _plot_reliability(calibration, output_dir / "reliability_diagram.png")
    _qualitative_gallery(predictions, manifest, config.raw_dir, output_dir)
    if robustness:
        _robustness_sweep(model, config, manifest, split, output_dir)
    console.print(f"[green]Evaluation complete:[/] {output_dir}")
    return output_dir


def build_leaderboard(runs_dir: str | Path = PROJECT_ROOT / "artifacts" / "runs") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    locked_experiments = {path.stem for path in (PROJECT_ROOT / "configs").glob("e*.yaml")}
    for run_dir in sorted(Path(runs_dir).glob("e*")):
        if run_dir.name not in locked_experiments:
            continue
        metrics_path = run_dir / "validation_metrics.json"
        config_path = run_dir / "config_resolved.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        import json

        with metrics_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        config = load_config(config_path)
        records.append(
            {
                "experiment_id": config.experiment_id,
                "features": config.features.kind,
                "model": config.model.name,
                "augmentation": config.training.augment,
                **metrics,
            }
        )
    if not records:
        raise ProjectError("No completed validation runs found")
    frame = pd.DataFrame(records).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    output_dir = ensure_directory(PROJECT_ROOT / "artifacts" / "tables")
    frame.to_csv(output_dir / "experiment_leaderboard.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_frame = frame.melt(
        id_vars="experiment_id", value_vars=["accuracy", "macro_f1"], var_name="metric"
    )
    sns.barplot(data=plot_frame, x="experiment_id", y="value", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Validation score")
    axis.set_title("Controlled validation comparison")
    figure.savefig(PROJECT_ROOT / "artifacts" / "figures" / "experiment_leaderboard.png", dpi=180)
    plt.close(figure)
    return frame


def promote_best_model() -> tuple[Path, Path]:
    leaderboard = build_leaderboard()
    best_id = str(leaderboard.iloc[0]["experiment_id"])
    source_run = PROJECT_ROOT / "artifacts" / "runs" / best_id
    model_destination = PROJECT_ROOT / "artifacts" / "models" / "speech_commands_best.keras"
    config_destination = PROJECT_ROOT / "artifacts" / "models" / "speech_commands_best.yaml"
    ensure_directory(model_destination.parent)
    shutil.copy2(source_run / "best_model.keras", model_destination)
    shutil.copy2(source_run / "config_resolved.yaml", config_destination)
    write_json(
        PROJECT_ROOT / "artifacts" / "models" / "selection.json",
        {
            "selected_experiment": best_id,
            "criterion": "validation macro-F1; validation accuracy tie-breaker",
            "validation_metrics": leaderboard.iloc[0].to_dict(),
        },
    )
    return model_destination, source_run
