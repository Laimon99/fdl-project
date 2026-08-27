from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from fdl_speech_commands.constants import (
    BACKGROUND_NOISE_DIR,
    LABEL_TO_INDEX,
    LABELS,
    TARGET_WORDS,
)
from fdl_speech_commands.data import (
    _interval_sets_overlap,
    speaker_id_from_filename,
    validate_manifest,
)
from fdl_speech_commands.datasets import load_noise_bank
from fdl_speech_commands.utils import ProjectError


def test_locked_label_space() -> None:
    assert len(LABELS) == 12
    assert LABELS[:2] == ("_silence_", "_unknown_")
    assert len(TARGET_WORDS) == 10


def test_speaker_id_removes_recording_suffix() -> None:
    assert speaker_id_from_filename("speaker123_nohash_4.wav") == "speaker123"
    assert speaker_id_from_filename("speaker123.wav") == "speaker123"


def test_interval_overlap_detection() -> None:
    assert _interval_sets_overlap([(0, 100), (200, 300)], [(99, 150)])
    assert not _interval_sets_overlap([(0, 100)], [(100, 200)])
    assert not _interval_sets_overlap([], [(0, 100)])


def test_validate_manifest_rejects_speaker_overlap(tmp_path: Path) -> None:
    (tmp_path / "dummy.wav").touch()
    rows = []
    for split in ("training", "validation", "testing"):
        for index, label in enumerate(LABELS):
            speaker = f"speaker-{split}"
            if split == "testing" and index == 0:
                speaker = "speaker-training"
            rows.append(
                {
                    "example_id": f"{split}-{index}",
                    "path": "dummy.wav",
                    "original_word": label,
                    "speaker_id": speaker,
                    "split": split,
                    "label": label,
                    "label_index": LABEL_TO_INDEX[label],
                    "source_type": "recording",
                    "offset_samples": 0,
                }
            )

    with pytest.raises(ProjectError, match="Speaker leakage between training and testing"):
        validate_manifest(pd.DataFrame(rows), tmp_path)


def test_noise_bank_uses_only_the_requested_temporal_region(tmp_path: Path) -> None:
    noise_dir = tmp_path / BACKGROUND_NOISE_DIR
    noise_dir.mkdir()
    waveform = np.linspace(-0.9, 0.9, 1_000, endpoint=False, dtype=np.float32)
    sf.write(noise_dir / "ordered.wav", waveform, 16_000)

    training = load_noise_bank(tmp_path, "training", clip_samples=100).numpy()
    validation = load_noise_bank(tmp_path, "validation", clip_samples=100).numpy()
    testing = load_noise_bank(tmp_path, "testing", clip_samples=100).numpy()

    assert training.shape == (8, 100)
    assert validation.shape == (1, 100)
    assert testing.shape == (1, 100)
    assert training.max() < validation.min()
    assert validation.max() < testing.min()
