from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from matplotlib import pyplot as plt
from rich.console import Console

from .constants import (
    BACKGROUND_NOISE_DIR,
    INVENTORY_PATH,
    LABELS,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    TARGET_WORDS,
)
from .data import build_raw_inventory, validate_manifest
from .utils import ProjectError, ensure_directory, write_json

console = Console()


def _rms_for_path(path: Path) -> float:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _plot_raw_word_counts(inventory: pd.DataFrame, output: Path) -> None:
    speech = inventory[inventory["original_word"] != BACKGROUND_NOISE_DIR]
    counts = speech["original_word"].value_counts().sort_values(ascending=True)
    colors = ["#2563eb" if word in TARGET_WORDS else "#94a3b8" for word in counts.index]
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    axis.barh(counts.index, counts.values, color=colors)
    axis.set_title("Speech Commands v0.01: recordings per original word")
    axis.set_xlabel("Recordings")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.99,
        0.02,
        "Blue = target command; grey = candidate for unknown",
        transform=axis.transAxes,
        ha="right",
        color="#475569",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_manifest_distribution(manifest: pd.DataFrame, output: Path) -> None:
    counts = (
        manifest.groupby(["label", "split"], observed=True)
        .size()
        .rename("examples")
        .reset_index()
    )
    counts["label"] = pd.Categorical(counts["label"], categories=LABELS, ordered=True)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=counts, x="label", y="examples", hue="split", ax=axis)
    axis.set_title("Locked 12-class modeling protocol")
    axis.set_xlabel("Class")
    axis.set_ylabel("Examples")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_duration_and_energy(
    inventory: pd.DataFrame,
    raw_dir: Path,
    output: Path,
    seed: int,
) -> pd.DataFrame:
    speech = inventory[inventory["original_word"] != BACKGROUND_NOISE_DIR].copy()
    sample_size = min(1_500, len(speech))
    sample = speech.sample(sample_size, random_state=seed).copy()
    sample["rms"] = [_rms_for_path(raw_dir / path) for path in sample["path"]]
    sample["target_status"] = np.where(
        sample["original_word"].isin(TARGET_WORDS), "target command", "unknown candidate"
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.histplot(
        data=speech,
        x="duration_seconds",
        bins=50,
        hue=np.where(speech["original_word"].isin(TARGET_WORDS), "target", "unknown"),
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set_title("Clip duration distribution")
    axes[0].set_xlabel("Seconds")
    axes[0].set_xlim(0, 1.05)
    sns.histplot(
        data=sample,
        x="rms",
        hue="target_status",
        bins=45,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set_title(f"Signal energy (stratified random audit, n={sample_size:,})")
    axes[1].set_xlabel("RMS amplitude")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return sample


def _load_manifest_clip(row: pd.Series, raw_dir: Path) -> np.ndarray:
    audio, sample_rate = sf.read(raw_dir / row["path"], dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    offset = int(row["offset_samples"])
    if row["source_type"] == "background_slice":
        audio = audio[offset : offset + SAMPLE_RATE]
    else:
        audio = audio[:SAMPLE_RATE]
    return np.pad(audio, (0, max(0, SAMPLE_RATE - len(audio))))[:SAMPLE_RATE]


def _plot_class_examples(manifest: pd.DataFrame, raw_dir: Path, output: Path) -> None:
    examples = (
        manifest[manifest["split"] == "training"]
        .groupby("label", observed=True, sort=False)
        .first()
        .reindex(LABELS)
    )
    figure, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for axis, (label, row) in zip(axes.flat, examples.iterrows(), strict=True):
        clip = _load_manifest_clip(row, raw_dir)
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
        axis.imshow(librosa.power_to_db(mel, ref=np.max), origin="lower", aspect="auto", cmap="magma")
        axis.set_title(label)
        axis.set_xlabel("Time frame")
        axis.set_ylabel("Mel bin")
    figure.suptitle("One training log-Mel spectrogram per modeled class", fontsize=17)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_eda(
    inventory_path: str | Path = INVENTORY_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    raw_dir: str | Path = RAW_DATA_DIR,
    seed: int = 42,
) -> Path:
    inventory_path, manifest_path, raw_dir = map(Path, (inventory_path, manifest_path, raw_dir))
    if not inventory_path.exists():
        inventory = build_raw_inventory(raw_dir, inventory_path)
    else:
        inventory = pd.read_csv(inventory_path, keep_default_na=False)
    if not manifest_path.exists():
        raise ProjectError("Modeling manifest is missing. Run `fdl-speech prepare` first.")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    validate_manifest(manifest, raw_dir)

    figure_dir = ensure_directory(PROJECT_ROOT / "artifacts" / "figures" / "eda")
    table_dir = ensure_directory(PROJECT_ROOT / "artifacts" / "tables")
    _plot_raw_word_counts(inventory, figure_dir / "raw_word_distribution.png")
    _plot_manifest_distribution(manifest, figure_dir / "modeling_distribution.png")
    energy_sample = _plot_duration_and_energy(
        inventory, raw_dir, figure_dir / "duration_and_energy.png", seed
    )
    _plot_class_examples(manifest, raw_dir, figure_dir / "class_spectrogram_examples.png")

    raw_counts = inventory["original_word"].value_counts().rename_axis("word").rename("recordings")
    raw_counts.to_csv(table_dir / "raw_word_counts.csv")
    split_counts = (
        manifest.groupby(["split", "label"], observed=True)
        .size()
        .rename("examples")
        .reset_index()
    )
    split_counts.to_csv(table_dir / "modeling_split_counts.csv", index=False)
    energy_sample.to_csv(table_dir / "eda_energy_sample.csv", index=False)

    speech = inventory[inventory["original_word"] != BACKGROUND_NOISE_DIR]
    speakers = speech["speaker_id"].nunique()
    summary: dict[str, Any] = {
        "raw_speech_recordings": len(speech),
        "background_recordings": int((inventory["original_word"] == BACKGROUND_NOISE_DIR).sum()),
        "original_words": int(speech["original_word"].nunique()),
        "unique_speakers": int(speakers),
        "sample_rate": sorted(inventory["sample_rate"].unique().tolist()),
        "channels": sorted(inventory["channels"].unique().tolist()),
        "duration_seconds": {
            "min": float(speech["duration_seconds"].min()),
            "median": float(speech["duration_seconds"].median()),
            "max": float(speech["duration_seconds"].max()),
        },
        "modeling_examples": len(manifest),
        "modeling_split_counts": manifest["split"].value_counts().to_dict(),
        "speaker_overlap_checked": True,
    }
    write_json(table_dir / "eda_summary.json", summary)
    console.print(f"[green]EDA artifacts written:[/] {figure_dir}")
    return figure_dir
