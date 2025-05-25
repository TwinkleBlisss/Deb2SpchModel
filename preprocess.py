"""I/O validation and ffmpeg‑based pre‑processing utilities."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from flask import abort

from config import (
    AUDIO_EXTS,
    TARGET_CHANNELS,
    TARGET_SR,
    MAX_DURATION,
)

__all__ = ["check_audio_file", "ffmpeg_convert"]


def check_audio_file(path: Path) -> None:
    """Validate that *path* exists and has an audio extension."""
    if not path.exists():
        abort(400, f"File not found: {path}")
    if path.suffix.lower() not in AUDIO_EXTS:
        abort(400, f"Unsupported file extension: {path.suffix}")


def ffmpeg_convert(src: Path) -> Path:
    """Convert *src* to mono/8 kHz WAV and truncate to ≤10 s.

    Returns a temporary WAV alongside *src* (same directory).
    """
    tmp_path = src.with_suffix(".mono8k.wav")
    cmd = [
        "ffmpeg",
        "-y",               # overwrite silently
        "-i",
        str(src),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SR),
        "-t",
        str(MAX_DURATION),
        str(tmp_path),
    ]
    logging.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        logging.exception("ffmpeg failed: %s", exc)
        abort(500, "Audio pre‑processing failed (ffmpeg error)")
    return tmp_path
