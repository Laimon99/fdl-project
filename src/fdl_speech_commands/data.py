from __future__ import annotations

import math
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn

from .constants import (
    BACKGROUND_NOISE_DIR,
    BACKGROUND_SPLIT_FRACTIONS,
    CLIP_SAMPLES,
    DATASET_ARCHIVE,
    DATASET_URL,
    DATASET_VERSION,
    INVENTORY_PATH,
    LABEL_TO_INDEX,
    LABELS,
    MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    SILENCE_LABEL,
    SILENCE_PERCENTAGE,
    SPLITS,
    TARGET_WORDS,
    UNKNOWN_LABEL,
    UNKNOWN_PERCENTAGE,
)
from .utils import ProjectError, ensure_directory, sha256_file, write_json

console = Console()


def download_dataset(force: bool = False) -> Path:
    """Download and safely extract Speech Commands v0.01."""
    if RAW_DATA_DIR.joinpath("validation_list.txt").exists() and not force:
        console.print(f"[green]Dataset already extracted:[/] {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    ensure_directory(DATASET_ARCHIVE.parent)
    ensure_directory(RAW_DATA_DIR.parent)
    if force and DATASET_ARCHIVE.exists():
        DATASET_ARCHIVE.unlink()

    if not DATASET_ARCHIVE.exists():
        _stream_download(DATASET_URL, DATASET_ARCHIVE)
    else:
        console.print(f"[green]Archive already downloaded:[/] {DATASET_ARCHIVE}")

    if RAW_DATA_DIR.exists() and force:
        shutil.rmtree(RAW_DATA_DIR)
    ensure_directory(RAW_DATA_DIR)
    _safe_extract_tar(DATASET_ARCHIVE, RAW_DATA_DIR)

    required = ["validation_list.txt", "testing_list.txt", "LICENSE"]
    missing = [name for name in required if not RAW_DATA_DIR.joinpath(name).exists()]
    if missing:
        raise ProjectError(f"Dataset extraction incomplete; missing {missing}")

    archive_info = {
        "version": DATASET_VERSION,
        "url": DATASET_URL,
        "archive_bytes": DATASET_ARCHIVE.stat().st_size,
        "archive_sha256": sha256_file(DATASET_ARCHIVE),
    }
    write_json(PROCESSED_DATA_DIR / "dataset_source.json", archive_info)
    console.print(f"[green]Extracted dataset to[/] {RAW_DATA_DIR}")
    return RAW_DATA_DIR


def _stream_download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    resume_from = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    response = requests.get(url, headers=headers, stream=True, timeout=(20, 120))
    response.raise_for_status()
    if resume_from and response.status_code != 206:
        resume_from = 0
        temporary.unlink(missing_ok=True)
    total = int(response.headers.get("Content-Length", "0")) + resume_from
    mode = "ab" if resume_from else "wb"

    columns = [
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*columns) as progress:
        task = progress.add_task(destination.name, total=total, completed=resume_from)
        with temporary.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    progress.update(task, advance=len(chunk))
    temporary.replace(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ProjectError(f"Unsafe path in dataset archive: {member.name}")
        tar.extractall(destination, filter="data")


def _read_official_list(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as stream:
        return {line.strip().replace("\\", "/") for line in stream if line.strip()}


def speaker_id_from_filename(filename: str | Path) -> str:
    stem = Path(filename).stem
    return stem.split("_nohash_", maxsplit=1)[0]


def _inspect_wav(path: Path, raw_dir: Path, validation: set[str], testing: set[str]) -> dict[str, Any]:
    relative = path.relative_to(raw_dir).as_posix()
    original_word = path.parent.name
    info = sf.info(path)
    if relative in validation:
        split = "validation"
    elif relative in testing:
        split = "testing"
    else:
        split = "training"
    return {
        "path": relative,
        "original_word": original_word,
        "speaker_id": "" if original_word == BACKGROUND_NOISE_DIR else speaker_id_from_filename(path),
        "split": split,
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
        "bytes": path.stat().st_size,
    }


def build_raw_inventory(raw_dir: Path = RAW_DATA_DIR, output: Path = INVENTORY_PATH) -> pd.DataFrame:
    """Index all WAV files, including non-target words and background noise."""
    if not raw_dir.exists():
        raise ProjectError("Dataset is missing. Run `fdl-speech download` first.")
    validation = _read_official_list(raw_dir / "validation_list.txt")
    testing = _read_official_list(raw_dir / "testing_list.txt")
    wav_paths = sorted(raw_dir.glob("*/*.wav"))
    if not wav_paths:
        raise ProjectError(f"No WAV files found below {raw_dir}")

    console.print(f"Inspecting {len(wav_paths):,} WAV headers...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        records = list(
            executor.map(
                lambda path: _inspect_wav(path, raw_dir, validation, testing),
                wav_paths,
            )
        )
    frame = pd.DataFrame.from_records(records).sort_values("path").reset_index(drop=True)
    ensure_directory(output.parent)
    frame.to_csv(output, index=False)
    console.print(f"[green]Raw inventory written:[/] {output}")
    return frame


def _silence_records(
    noise_inventory: pd.DataFrame,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create virtual background-only clips from split-specific temporal regions."""
    start_fraction, end_fraction = BACKGROUND_SPLIT_FRACTIONS[split]
    sources = noise_inventory.sort_values("path").to_dict("records")
    if not sources:
        raise ProjectError("The dataset does not contain background-noise recordings")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for index in range(count):
        source = sources[index % len(sources)]
        total_frames = int(source["frames"])
        region_start = math.ceil(total_frames * start_fraction)
        region_end = math.floor(total_frames * end_fraction)
        latest_start = region_end - CLIP_SAMPLES
        if latest_start < region_start:
            raise ProjectError(
                f"Background recording {source['path']} is too short for split {split}"
            )
        offset = int(rng.integers(region_start, latest_start + 1))
        records.append(
            {
                "path": source["path"],
                "original_word": BACKGROUND_NOISE_DIR,
                "speaker_id": "",
                "split": split,
                "label": SILENCE_LABEL,
                "label_index": LABEL_TO_INDEX[SILENCE_LABEL],
                "source_type": "background_slice",
                "offset_samples": offset,
                "frames": CLIP_SAMPLES,
                "sample_rate": SAMPLE_RATE,
                "is_synthetic": True,
            }
        )
    return records


def build_manifest(
    raw_dir: Path = RAW_DATA_DIR,
    inventory_path: Path = INVENTORY_PATH,
    output: Path = MANIFEST_PATH,
    seed: int = 42,
    unknown_percentage: float = UNKNOWN_PERCENTAGE,
    silence_percentage: float = SILENCE_PERCENTAGE,
) -> pd.DataFrame:
    """Build the deterministic 12-class manifest used by every experiment."""
    inventory = (
        pd.read_csv(inventory_path, keep_default_na=False)
        if inventory_path.exists()
        else build_raw_inventory(raw_dir, inventory_path)
    )
    if set(inventory["sample_rate"].unique()) != {SAMPLE_RATE}:
        raise ProjectError("Expected every Speech Commands recording to use 16 kHz audio")
    if set(inventory["channels"].unique()) != {1}:
        raise ProjectError("Expected every Speech Commands recording to be mono")

    speech = inventory[inventory["original_word"] != BACKGROUND_NOISE_DIR].copy()
    noise = inventory[inventory["original_word"] == BACKGROUND_NOISE_DIR].copy()
    speech["label"] = np.where(
        speech["original_word"].isin(TARGET_WORDS),
        speech["original_word"],
        UNKNOWN_LABEL,
    )

    split_seed = {"training": seed + 11, "validation": seed + 23, "testing": seed + 37}
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        split_rows = speech[speech["split"] == split]
        wanted = split_rows[split_rows["label"] != UNKNOWN_LABEL].copy()
        unknown_pool = split_rows[split_rows["label"] == UNKNOWN_LABEL].copy()
        unknown_count = math.ceil(len(wanted) * unknown_percentage / 100)
        silence_count = math.ceil(len(wanted) * silence_percentage / 100)
        if unknown_count > len(unknown_pool):
            raise ProjectError(f"Not enough unknown examples in {split}")
        unknown = unknown_pool.sample(n=unknown_count, random_state=split_seed[split])
        selected = pd.concat([wanted, unknown], ignore_index=True)
        for row in selected.to_dict("records"):
            label = str(row["label"])
            records.append(
                {
                    "path": row["path"],
                    "original_word": row["original_word"],
                    "speaker_id": row["speaker_id"],
                    "split": split,
                    "label": label,
                    "label_index": LABEL_TO_INDEX[label],
                    "source_type": "recording",
                    "offset_samples": 0,
                    "frames": int(row["frames"]),
                    "sample_rate": int(row["sample_rate"]),
                    "is_synthetic": False,
                }
            )
        records.extend(_silence_records(noise, split, silence_count, split_seed[split] + 101))

    manifest = pd.DataFrame.from_records(records)
    manifest.insert(0, "example_id", [f"scv1-{index:06d}" for index in range(len(manifest))])
    split_order = pd.Categorical(manifest["split"], categories=SPLITS, ordered=True)
    manifest = (
        manifest.assign(_split_order=split_order)
        .sort_values(["_split_order", "label_index", "path", "offset_samples"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )
    manifest["example_id"] = [f"scv1-{index:06d}" for index in range(len(manifest))]
    validate_manifest(manifest, raw_dir)
    ensure_directory(output.parent)
    manifest.to_csv(output, index=False)

    counts = (
        manifest.groupby(["split", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, columns=LABELS, fill_value=0)
    )
    metadata = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "unknown_percentage": unknown_percentage,
        "silence_percentage": silence_percentage,
        "labels": list(LABELS),
        "examples": len(manifest),
        "counts": counts.to_dict(orient="index"),
        "manifest_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    console.print(f"[green]Modeling manifest written:[/] {output} ({len(manifest):,} rows)")
    return manifest


def validate_manifest(manifest: pd.DataFrame, raw_dir: Path = RAW_DATA_DIR) -> None:
    required = {
        "example_id",
        "path",
        "original_word",
        "speaker_id",
        "split",
        "label",
        "label_index",
        "source_type",
        "offset_samples",
    }
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ProjectError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if manifest["example_id"].duplicated().any():
        raise ProjectError("Manifest example IDs are not unique")
    if set(manifest["split"]) != set(SPLITS):
        raise ProjectError("Manifest does not contain all required splits")
    if set(manifest["label"]) != set(LABELS):
        raise ProjectError("Manifest does not contain all 12 labels")
    mapped = manifest["label"].map(LABEL_TO_INDEX)
    if not np.array_equal(mapped.to_numpy(), manifest["label_index"].to_numpy()):
        raise ProjectError("Manifest label indices do not match the locked label order")

    real = manifest[manifest["source_type"] == "recording"]
    speakers = {
        split: set(real.loc[real["split"] == split, "speaker_id"])
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = speakers[left] & speakers[right]
            if overlap:
                raise ProjectError(
                    f"Speaker leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )

    missing_paths = [path for path in manifest["path"].unique() if not (raw_dir / path).is_file()]
    if missing_paths:
        raise ProjectError(f"Manifest references missing audio: {missing_paths[:5]}")

    synthetic = manifest[manifest["source_type"] == "background_slice"]
    intervals: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for row in synthetic.itertuples(index=False):
        intervals.setdefault(row.path, {}).setdefault(row.split, []).append(
            (int(row.offset_samples), int(row.offset_samples) + CLIP_SAMPLES)
        )
    for path, by_split in intervals.items():
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                if _interval_sets_overlap(by_split.get(left, []), by_split.get(right, [])):
                    raise ProjectError(f"Silence source overlap across splits in {path}")


def _interval_sets_overlap(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> bool:
    if not left or not right:
        return False
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    left_index = right_index = 0
    while left_index < len(left_sorted) and right_index < len(right_sorted):
        left_start, left_end = left_sorted[left_index]
        right_start, right_end = right_sorted[right_index]
        if max(left_start, right_start) < min(left_end, right_end):
            return True
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return False


def prepare_data(force_inventory: bool = False, seed: int = 42) -> pd.DataFrame:
    if force_inventory or not INVENTORY_PATH.exists():
        build_raw_inventory()
    return build_manifest(seed=seed)
