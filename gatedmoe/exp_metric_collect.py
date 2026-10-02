"""Metric collection and aggregation across experiment runs."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from gatedmoe.util_config_io import LOG_DIR


def load_all_results(log_dir: str = str(LOG_DIR), prefix: str = "") -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    rows = []
    log_path = Path(log_dir)
    for exp_dir in sorted(log_path.glob(f"{prefix}*")):
        result_file = exp_dir / "results.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            data = json.load(f)
        cfg = data["config"]
        last_epoch = data["epochs"][-1] if data["epochs"] else {}
        rows.append({
            "experiment": exp_dir.name,
            "method": cfg.get("method", ""),
            "budget_kb": cfg.get("budget_bytes", 0) / 1024,
            "num_experts": cfg.get("num_experts", 0),
            "seed": cfg.get("seed", 0),
            "best_test_acc": data.get("best_test_acc", 0),
            "final_test_acc": data.get("final_test_acc", 0),
            "peak_sram_kb": last_epoch.get("peak_sram_bytes", 0) / 1024,
            "flash_to_sram_kb": last_epoch.get("flash_to_sram_bytes", 0) / 1024,
            "avg_macs": last_epoch.get("avg_macs_per_step", 0),
        })
    return pd.DataFrame(rows)


def aggregate_by_method_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std across seeds for each (method, budget) pair."""
    grouped = df.groupby(["method", "budget_kb"]).agg(
        acc_mean=("best_test_acc", "mean"),
        acc_std=("best_test_acc", "std"),
        sram_mean=("peak_sram_kb", "mean"),
        flash_mean=("flash_to_sram_kb", "mean"),
        macs_mean=("avg_macs", "mean"),
    ).reset_index()
    return grouped


def aggregate_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for ablation experiments."""
    grouped = df.groupby(["method", "num_experts"]).agg(
        acc_mean=("best_test_acc", "mean"),
        acc_std=("best_test_acc", "std"),
        sram_mean=("peak_sram_kb", "mean"),
    ).reset_index()
    return grouped
