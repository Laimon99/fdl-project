from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class ProjectError(RuntimeError):
    """Raised for actionable project-level failures."""


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def set_global_determinism(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        with suppress(AttributeError, RuntimeError):
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_revision": git_revision(),
    }
    try:
        import tensorflow as tf

        metadata.update(
            {
                "tensorflow": tf.__version__,
                "keras": tf.keras.__version__,
                "devices": [device.name for device in tf.config.list_physical_devices()],
            }
        )
    except ImportError:
        metadata["tensorflow"] = None
    return metadata
