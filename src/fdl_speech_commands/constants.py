from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_VERSION = "v0.01"
DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.01.tar.gz"
)
DATASET_ARCHIVE = PROJECT_ROOT / "data" / "downloads" / "speech_commands_v0.01.tar.gz"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "speech_commands_v0.01"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DATA_DIR / "manifest.csv"
INVENTORY_PATH = PROCESSED_DATA_DIR / "raw_inventory.csv"

TARGET_WORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
LABELS = (SILENCE_LABEL, UNKNOWN_LABEL, *TARGET_WORDS)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

BACKGROUND_NOISE_DIR = "_background_noise_"
SAMPLE_RATE = 16_000
CLIP_SAMPLES = 16_000
DEFAULT_SEED = 42
UNKNOWN_PERCENTAGE = 10.0
SILENCE_PERCENTAGE = 10.0
SPLITS = ("training", "validation", "testing")

