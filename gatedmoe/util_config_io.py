"""Configuration loading, GPG encryption helpers, and path utilities."""

import os
import yaml
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Project paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
FIGURE_DIR = RESULT_DIR / "figure"
LOG_DIR = RESULT_DIR / "logs"
TABLE_DIR = RESULT_DIR / "tables"
CONFIG_DIR = PROJECT_ROOT / "configs"

for d in [FIGURE_DIR, LOG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Experiment config ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model: int = 128
    d_ff_shared: int = 256
    d_ff_expert: object = None  # int or list; None = use default [128, 256, 512, 1024]
    num_experts: int = 4
    num_layers: int = 4
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 32
    patch_size: int = 4
    admissible_mask: object = None  # list[int] expert ids (S1 choice-set study)
    capacity_factor: float = 1.25   # E5 token_capacity control (tuned on val)


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class MemoryConfig:
    budget_bytes: int = 8 * 1024 * 1024  # 8 MB default
    dtype_bytes: int = 4            # float32


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    method: str = "gatedmoe_g"
    dataset: str = "cifar10"  # cifar10 or cifar100
    experiment_name: str = "default"
    save_dir: str = str(LOG_DIR)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig()
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "memory" in raw:
        cfg.memory = MemoryConfig(**raw["memory"])
    for key in ("method", "experiment_name", "save_dir"):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


def save_config(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
