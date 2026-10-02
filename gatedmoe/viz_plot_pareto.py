"""Pareto curve and main result figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from gatedmoe.util_config_io import FIGURE_DIR
from gatedmoe.exp_metric_collect import load_all_results, aggregate_by_method_budget


METHOD_STYLES = {
    "dense":         {"color": "#888888", "marker": "s", "label": "Dense (all experts)"},
    "learned_top1":  {"color": "#e74c3c", "marker": "^", "label": "Learned Top-1"},
    "random":        {"color": "#f39c12", "marker": "v", "label": "Random"},
    "round_robin":   {"color": "#3498db", "marker": "D", "label": "Round-Robin"},
    "mod":           {"color": "#9b59b6", "marker": "p", "label": "Mixture-of-Depths"},
    "gatedmoe_base": {"color": "#2ecc71", "marker": "o", "label": "GatedMoE-Base"},
    "gatedmoe_g":    {"color": "#e91e63", "marker": "*", "label": "GatedMoE-G"},
}


def plot_pareto_curve(log_dir: str = None, save_path: str = None):
    """Hero figure: accuracy vs peak SRAM for all methods."""
    df = load_all_results(log_dir or "", prefix="sweep_")
    if df.empty:
        print("No sweep results found.")
        return
    agg = aggregate_by_method_budget(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method, style in METHOD_STYLES.items():
        subset = agg[agg["method"] == method].sort_values("sram_mean")
        if subset.empty:
            continue
        ax.plot(
            subset["sram_mean"], subset["acc_mean"],
            color=style["color"], marker=style["marker"],
            label=style["label"], linewidth=2, markersize=8,
        )

    ax.set_xlabel("Peak SRAM (KB)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Peak SRAM Budget", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = save_path or str(FIGURE_DIR / "pareto-accuracy-vs-sram.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_flash_traffic(log_dir: str = None, save_path: str = None):
    """Flash→SRAM data movement comparison."""
    df = load_all_results(log_dir or "", prefix="sweep_")
    if df.empty:
        return
    agg = aggregate_by_method_budget(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method, style in METHOD_STYLES.items():
        subset = agg[agg["method"] == method].sort_values("sram_mean")
        if subset.empty:
            continue
        ax.plot(
            subset["sram_mean"], subset["flash_mean"],
            color=style["color"], marker=style["marker"],
            label=style["label"], linewidth=2, markersize=8,
        )

    ax.set_xlabel("Peak SRAM (KB)", fontsize=12)
    ax.set_ylabel("Flash→SRAM Traffic (KB)", fontsize=12)
    ax.set_title("Data Movement vs. SRAM Budget", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = save_path or str(FIGURE_DIR / "flash-traffic-vs-sram.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
