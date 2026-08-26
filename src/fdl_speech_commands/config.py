from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import PROJECT_ROOT


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "log_mel"
    sample_rate: int = 16_000
    clip_samples: int = 16_000
    frame_length: int = 480
    frame_step: int = 160
    fft_length: int = 512
    mel_bins: int = 40
    mfcc_bins: int = 13
    lower_hertz: float = 20.0
    upper_hertz: float = 7_600.0

    def validate(self) -> None:
        if self.kind not in {"log_mel", "mfcc"}:
            raise ValueError(f"Unsupported feature kind: {self.kind}")
        if self.sample_rate <= 0 or self.clip_samples <= 0:
            raise ValueError("Sample rate and clip length must be positive")
        if not 0 <= self.lower_hertz < self.upper_hertz <= self.sample_rate / 2:
            raise ValueError("Mel frequency bounds must lie inside the Nyquist interval")
        if self.mfcc_bins > self.mel_bins:
            raise ValueError("mfcc_bins cannot exceed mel_bins")


@dataclass(frozen=True)
class AugmentationConfig:
    max_shift_ms: int = 100
    gain_min: float = 0.7
    gain_max: float = 1.3
    noise_probability: float = 0.8
    snr_db_min: float = 5.0
    snr_db_max: float = 25.0
    time_masks: int = 2
    time_mask_max: int = 10
    frequency_masks: int = 2
    frequency_mask_max: int = 5

    def validate(self) -> None:
        if self.max_shift_ms < 0:
            raise ValueError("max_shift_ms must be non-negative")
        if not 0 < self.gain_min <= self.gain_max:
            raise ValueError("Invalid gain interval")
        if not 0 <= self.noise_probability <= 1:
            raise ValueError("noise_probability must be in [0, 1]")
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("Invalid SNR interval")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "small_cnn"
    dropout: float = 0.3

    def validate(self) -> None:
        if self.name not in {"mlp", "small_cnn", "ds_cnn", "crnn"}:
            raise ValueError(f"Unsupported model: {self.name}")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    augment: bool = False

    def validate(self) -> None:
        if min(self.batch_size, self.epochs, self.patience) <= 0:
            raise ValueError("batch_size, epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int
    manifest: Path
    raw_dir: Path
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    source_path: Path | None = None

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment.id cannot be empty")
        self.features.validate()
        self.model.validate()
        self.training.validate()
        self.augmentation.validate()
        if self.model.name == "mlp" and self.features.kind != "mfcc":
            raise ValueError("The MLP baseline is defined on MFCC features")

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": {"id": self.experiment_id, "seed": self.seed},
            "data": {"manifest": str(self.manifest), "raw_dir": str(self.raw_dir)},
            "features": vars(self.features),
            "model": vars(self.model),
            "training": vars(self.training),
            "augmentation": vars(self.augmentation),
        }


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    experiment = raw.get("experiment", {})
    data = raw.get("data", {})
    config = ExperimentConfig(
        experiment_id=str(experiment["id"]),
        seed=int(experiment.get("seed", 42)),
        manifest=_resolve_project_path(data["manifest"]),
        raw_dir=_resolve_project_path(data["raw_dir"]),
        features=FeatureConfig(**raw.get("features", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
        source_path=source,
    )
    config.validate()
    return config

