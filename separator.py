"""Lazy loader/wrapper around MossFormer2Wrapper for inference only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch

from config import CONFIG_NAME, TARGET_SR

# Import model implementation from the cloned MossFormer2 repo
# Structure:
#   model/
#     mossformer2.py
#     mossformer2_configs.py
#     utils/
#       ...
try:
    from model.mossformer2 import Mossformer2Wrapper
    from model import mossformer2_configs as mf_cfg
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Cannot import Mossformer2 modules."
    ) from exc


class _SeparatorSingleton:
    """Singleton wrapper that holds the heavy Torch model in memory."""

    _instance: "_SeparatorSingleton | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        # Grab config by attribute name or dict lookup
        if hasattr(mf_cfg, CONFIG_NAME):
            config = getattr(mf_cfg, CONFIG_NAME)
        elif isinstance(mf_cfg.__dict__.get("configs"), dict):
            config = mf_cfg.configs[CONFIG_NAME]
        else:
            raise RuntimeError(f"Config '{CONFIG_NAME}' not found in mossformer2_configs")

        logging.info("Loading MossFormer2 (%s)…", CONFIG_NAME)
        self.model = Mossformer2Wrapper(config)
        self.model.loadPretrained()
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Model ready on %s", self.device)

    def separate(self, mix: Path, out_dir: Path) -> List[Path]:
        """Run separation on *mix* (WAV) → returns list with paths to stems."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.inference(str(mix), str(out_dir))  # writes index1.wav, index2.wav
        paths = sorted(out_dir.glob("index*.wav"))
        if not paths:
            raise RuntimeError("Model did not produce output files")
        # Ensure correct sample‑rate for downstream consumers (normalised already)
        return [p.resolve() for p in paths]


# Export a process‑wide singleton
separator = _SeparatorSingleton()

__all__ = ["separator"]
