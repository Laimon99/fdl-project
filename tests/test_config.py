from pathlib import Path

import pytest

from fdl_speech_commands.config import FeatureConfig, load_config
from fdl_speech_commands.constants import PROJECT_ROOT


def test_every_experiment_config_is_valid() -> None:
    paths = sorted((PROJECT_ROOT / "configs").glob("e*.yaml"))
    assert len(paths) == 5
    identifiers = {load_config(path).experiment_id for path in paths}
    assert len(identifiers) == len(paths)


def test_invalid_frequency_bound_is_rejected() -> None:
    config = FeatureConfig(upper_hertz=9_000)
    with pytest.raises(ValueError, match="Nyquist"):
        config.validate()


def test_config_paths_are_project_relative() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "e01_mfcc_mlp.yaml")
    assert config.manifest == PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    assert config.source_path == Path(PROJECT_ROOT / "configs" / "e01_mfcc_mlp.yaml")
    assert config.as_dict()["data"]["manifest"] == "data/processed/manifest.csv"
