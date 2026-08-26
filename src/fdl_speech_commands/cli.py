from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Annotated

import numpy as np
import tensorflow as tf
import typer
from rich.console import Console
from rich.table import Table

from .config import FeatureConfig, ModelConfig, load_config
from .constants import (
    DATASET_VERSION,
    LABELS,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from .data import download_dataset, prepare_data, validate_manifest
from .datasets import read_manifest
from .eda import run_eda
from .evaluation import build_leaderboard, evaluate_run, promote_best_model
from .features import extract_features, feature_shape
from .inference import predict_file
from .models import build_model
from .training import train_all, train_experiment
from .utils import ProjectError, runtime_metadata, set_global_determinism

app = typer.Typer(
    name="fdl-speech",
    help="Reproducible Speech Commands v0.01 research pipeline.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def doctor() -> None:
    """Inspect software, devices, dataset, and repository readiness."""
    metadata = runtime_metadata()
    table = Table(title="Project environment")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("Platform", platform.platform())
    table.add_row("TensorFlow", str(metadata.get("tensorflow")))
    table.add_row("Keras", str(metadata.get("keras")))
    table.add_row("Physical devices", ", ".join(metadata.get("devices", [])))
    table.add_row("Dataset protocol", DATASET_VERSION)
    table.add_row("Raw dataset", "ready" if RAW_DATA_DIR.exists() else "not downloaded")
    table.add_row("Manifest", "ready" if MANIFEST_PATH.exists() else "not prepared")
    console.print(table)


@app.command("download")
def download_command(
    force: Annotated[bool, typer.Option(help="Redownload and re-extract the dataset.")] = False,
) -> None:
    """Download Speech Commands from the official TensorFlow URL."""
    download_dataset(force=force)


@app.command("prepare")
def prepare_command(
    force_inventory: Annotated[
        bool, typer.Option(help="Re-read every WAV header before manifest creation.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Deterministic manifest seed.")] = 42,
) -> None:
    """Create raw inventory and leakage-checked 12-class manifest."""
    prepare_data(force_inventory=force_inventory, seed=seed)


@app.command("eda")
def eda_command(seed: Annotated[int, typer.Option()] = 42) -> None:
    """Generate the committed exploratory analysis figures and tables."""
    run_eda(seed=seed)


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    overwrite: Annotated[bool, typer.Option(help="Replace an existing run directory.")] = False,
    epochs: Annotated[int | None, typer.Option(help="Override epochs for diagnostics.")] = None,
) -> None:
    """Train one configured experiment."""
    train_experiment(config, overwrite=overwrite, epochs_override=epochs)


@app.command("train-all")
def train_all_command(
    overwrite: Annotated[bool, typer.Option(help="Replace existing experiment runs.")] = False,
    epochs: Annotated[int | None, typer.Option(help="Override epochs for diagnostics.")] = None,
) -> None:
    """Train the complete locked experiment matrix."""
    train_all(overwrite=overwrite, epochs_override=epochs)


@app.command("evaluate")
def evaluate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    split: Annotated[str, typer.Option(help="validation or testing")] = "testing",
    bootstrap_resamples: Annotated[int, typer.Option(min=100)] = 10_000,
    robustness: Annotated[bool, typer.Option(help="Run deterministic corruption sweep.")] = True,
) -> None:
    """Run quantitative and qualitative evaluation of a frozen model."""
    if split not in {"validation", "testing"}:
        raise typer.BadParameter("split must be validation or testing")
    evaluate_run(run_dir, split, bootstrap_resamples, robustness)


@app.command("leaderboard")
def leaderboard_command() -> None:
    """Rank completed experiments using validation results only."""
    console.print(build_leaderboard().to_string(index=False))


@app.command("promote")
def promote_command() -> None:
    """Copy the validation-selected model to the final artifact location."""
    model, run_dir = promote_best_model()
    console.print(f"[green]Promoted[/] {run_dir.name} -> {model}")


@app.command("infer")
def infer_command(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model: Annotated[Path, typer.Option()] = PROJECT_ROOT
    / "artifacts"
    / "models"
    / "speech_commands_best.keras",
    config: Annotated[Path, typer.Option()] = PROJECT_ROOT
    / "artifacts"
    / "models"
    / "speech_commands_best.yaml",
    output_json: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Classify one WAV/audio file and print the top-three predictions."""
    console.print_json(json.dumps(predict_file(audio, model, config, output_json)))


@app.command("smoke-test")
def smoke_test() -> None:
    """Validate feature extraction and every model with synthetic audio."""
    set_global_determinism(42)
    for feature_kind, model_name in (
        ("mfcc", "mlp"),
        ("log_mel", "small_cnn"),
        ("log_mel", "ds_cnn"),
        ("log_mel", "crnn"),
    ):
        features_config = FeatureConfig(kind=feature_kind)
        waveform = tf.convert_to_tensor(np.zeros(16_000, dtype=np.float32))
        features = extract_features(waveform, features_config)
        model, normalizer = build_model(feature_shape(features_config), ModelConfig(model_name))
        normalizer.adapt(features[tf.newaxis])
        output = model(features[tf.newaxis], training=False)
        if tuple(output.shape) != (1, len(LABELS)) or not np.isfinite(output.numpy()).all():
            raise ProjectError(f"Smoke test failed for {model_name}")
        config_paths = list((PROJECT_ROOT / "configs").glob("e*.yaml"))
        for path in config_paths:
            load_config(path)
    if MANIFEST_PATH.exists():
        manifest = read_manifest(MANIFEST_PATH)
        validate_manifest(manifest)
    console.print("[green]Smoke test passed for all feature/model paths.[/]")


def main() -> None:
    try:
        app()
    except ProjectError as error:
        console.print(f"[bold red]Project error:[/] {error}")
        raise typer.Exit(code=1) from error

