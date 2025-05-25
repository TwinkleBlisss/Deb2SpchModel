"""Centralised configuration constants."""

from pathlib import Path

AUDIO_EXTS: set[str] = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SEPARATED_SUBDIR = "separated"
TARGET_SR = 8_000  # Hz
TARGET_CHANNELS = 1
MAX_DURATION = 10  # seconds

# Which pretrained checkpoint (see `model_weights/<config_name>`)
CONFIG_NAME = "mossformer2_librimix_2spk"

PROJECT_ROOT = Path(__file__).resolve().parent
