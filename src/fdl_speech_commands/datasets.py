from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from rich.console import Console

from .augmentation import augment_waveform, spec_augment
from .config import AugmentationConfig, FeatureConfig
from .constants import BACKGROUND_NOISE_DIR, CLIP_SAMPLES
from .features import extract_features
from .utils import ProjectError

console = Console()
_WAVEFORM_TENSOR_CACHE: dict[tuple[str, str, int], tf.Tensor] = {}


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, keep_default_na=False)
    expected = {"path", "split", "label", "label_index", "source_type", "offset_samples"}
    missing = expected - set(manifest.columns)
    if missing:
        raise ProjectError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest


def load_noise_bank(raw_dir: str | Path, clip_samples: int = CLIP_SAMPLES) -> tf.Tensor:
    clips: list[np.ndarray] = []
    for path in sorted(Path(raw_dir).joinpath(BACKGROUND_NOISE_DIR).glob("*.wav")):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != 16_000:
            raise ProjectError(f"Unexpected noise sample rate in {path}: {sample_rate}")
        if audio.ndim != 1:
            audio = np.mean(audio, axis=-1)
        for start in range(0, max(1, len(audio) - clip_samples + 1), clip_samples):
            clip = audio[start : start + clip_samples]
            if len(clip) == clip_samples:
                clips.append(clip)
    if not clips:
        raise ProjectError("No complete background-noise clips were found")
    return tf.convert_to_tensor(np.stack(clips), dtype=tf.float32)


def _decode_record(
    path: tf.Tensor,
    source_type: tf.Tensor,
    offset_samples: tf.Tensor,
    clip_samples: int,
) -> tf.Tensor:
    encoded = tf.io.read_file(path)
    decoded, _ = tf.audio.decode_wav(encoded, desired_channels=1)
    waveform = tf.squeeze(decoded, axis=-1)
    offset = tf.cast(offset_samples, tf.int32)
    waveform = tf.cond(
        tf.equal(source_type, "background_slice"),
        lambda: waveform[offset : offset + clip_samples],
        lambda: waveform[:clip_samples],
    )
    padding = tf.maximum(0, clip_samples - tf.shape(waveform)[0])
    waveform = tf.pad(waveform, [[0, padding]])[:clip_samples]
    return tf.ensure_shape(waveform, [clip_samples])


def _load_waveform_tensor(
    subset: pd.DataFrame,
    raw_dir: Path,
    split: str,
    clip_samples: int,
) -> tf.Tensor:
    """Load selected PCM16 clips once, avoiding repeated small-file I/O every epoch."""
    key = (str(raw_dir.resolve()), split, clip_samples)
    if key in _WAVEFORM_TENSOR_CACHE:
        return _WAVEFORM_TENSOR_CACHE[key]
    console.print(
        f"Caching {len(subset):,} {split} waveforms as PCM16 "
        f"({len(subset) * clip_samples * 2 / 2**20:.0f} MiB)..."
    )
    waveforms = np.zeros((len(subset), clip_samples), dtype=np.int16)
    for index, row in enumerate(subset.itertuples(index=False)):
        audio, sample_rate = sf.read(
            raw_dir / row.path,
            dtype="int16",
            always_2d=False,
        )
        if sample_rate != 16_000:
            raise ProjectError(f"Unexpected sample rate in {row.path}: {sample_rate}")
        if audio.ndim != 1:
            audio = np.mean(audio.astype(np.float32), axis=-1).astype(np.int16)
        offset = int(row.offset_samples) if row.source_type == "background_slice" else 0
        clip = audio[offset : offset + clip_samples]
        waveforms[index, : len(clip)] = clip
    tensor = tf.convert_to_tensor(waveforms)
    _WAVEFORM_TENSOR_CACHE[key] = tensor
    return tensor


def build_dataset(
    manifest: pd.DataFrame,
    raw_dir: str | Path,
    feature_config: FeatureConfig,
    augmentation_config: AugmentationConfig | None = None,
    split: Literal["training", "validation", "testing"] = "training",
    batch_size: int = 128,
    training: bool = False,
    seed: int = 42,
    include_metadata: bool = False,
    corruption_snr_db: float | None = None,
    time_shift_samples: int = 0,
    cache_waveforms: bool = True,
) -> tf.data.Dataset:
    subset = manifest.loc[manifest["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise ProjectError(f"Manifest split is empty: {split}")
    root = Path(raw_dir)
    paths = np.asarray([str(root / path) for path in subset["path"]], dtype=str)
    source_types = subset["source_type"].astype(str).to_numpy()
    offsets = subset["offset_samples"].astype(np.int64).to_numpy()
    labels = subset["label_index"].astype(np.int64).to_numpy()
    example_ids = subset["example_id"].astype(str).to_numpy()

    tensors = {
        "path": paths,
        "source_type": source_types,
        "offset_samples": offsets,
        "label": labels,
        "example_id": example_ids,
    }
    if cache_waveforms:
        tensors["waveform_int16"] = _load_waveform_tensor(
            subset, root, split, feature_config.clip_samples
        )
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    noise_bank = None
    if (training and augmentation_config is not None) or corruption_snr_db is not None:
        noise_bank = load_noise_bank(root, feature_config.clip_samples)

    def transform(record: dict[str, tf.Tensor]):
        if cache_waveforms:
            waveform = tf.cast(record["waveform_int16"], tf.float32) / 32_768.0
            waveform = tf.ensure_shape(waveform, [feature_config.clip_samples])
        else:
            waveform = _decode_record(
                record["path"],
                record["source_type"],
                record["offset_samples"],
                feature_config.clip_samples,
            )
        if training and augmentation_config is not None:
            waveform = augment_waveform(waveform, noise_bank, feature_config, augmentation_config)
        if time_shift_samples:
            shift = int(time_shift_samples)
            if shift > 0:
                waveform = tf.concat([tf.zeros([shift]), waveform[:-shift]], axis=0)
            else:
                amount = -shift
                waveform = tf.concat([waveform[amount:], tf.zeros([amount])], axis=0)
        if corruption_snr_db is not None:
            noise_index = tf.strings.to_hash_bucket_fast(
                record["example_id"], int(noise_bank.shape[0])
            )
            noise = noise_bank[noise_index]
            epsilon = tf.constant(1e-7, tf.float32)
            signal_rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)) + epsilon)
            noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise)) + epsilon)
            reference_rms = tf.maximum(signal_rms, tf.constant(0.025, tf.float32))
            scale = reference_rms / (
                noise_rms * tf.pow(10.0, tf.cast(corruption_snr_db, tf.float32) / 20.0)
            )
            waveform = tf.clip_by_value(waveform + noise * scale, -1.0, 1.0)
        features = extract_features(waveform, feature_config)
        if training and augmentation_config is not None:
            features = spec_augment(features, augmentation_config)
        if include_metadata:
            metadata = {
                "example_id": record["example_id"],
                "path": record["path"],
                "waveform": waveform,
            }
            return features, record["label"], metadata
        return features, record["label"]

    dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if augmentation_config is None and corruption_snr_db is None and time_shift_samples == 0:
        # Cache deterministic features, not augmented examples. Validation is consumed several
        # times per epoch and unaugmented training features are reused across all epochs.
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(len(subset), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
