"""Ablation and analysis figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from gatedmoe.util_config_io import FIGURE_DIR
from gatedmoe.exp_metric_collect import load_all_results, aggregate_ablation


def plot_ablation_heatmap(log_dir: str = None, save_path: str = None):
    """Ablation heatmap: debt type × num_experts → accuracy."""
    df = load_all_results(log_dir or "", prefix="ablation_")
    if df.empty:
        print("No ablation results found.")
        return
    agg = aggregate_ablation(df)

    methods = ["round_robin", "gatedmoe_base", "gatedmoe_g"]
    method_labels = ["No Debt (RR)", "GatedMoE-Base", "GatedMoE-G"]
    experts_list = sorted(agg["num_experts"].unique())

    matrix = np.zeros((len(methods), len(experts_list)))
    for i, m in enumerate(methods):
        for j, e in enumerate(experts_list):
            row = agg[(agg["method"] == m) & (agg["num_experts"] == e)]
            if not row.empty:
                matrix[i, j] = row["acc_mean"].values[0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=matrix.min() - 0.02)

    ax.set_xticks(range(len(experts_list)))
    ax.set_xticklabels([str(e) for e in experts_list])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(method_labels)
    ax.set_xlabel("Number of Experts")
    ax.set_ylabel("Routing Method")
    ax.set_title("Ablation: Accuracy by Debt Type × Experts")

    for i in range(len(methods)):
        for j in range(len(experts_list)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, label="Test Accuracy")
    plt.tight_layout()

    out = save_path or str(FIGURE_DIR / "ablation-heatmap.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_expert_activation_timeline(results_json: str, save_path: str = None):
    """Expert activation over training steps for one run."""
    import json
    with open(results_json) as f:
        data = json.load(f)

    num_layers = len(data["epochs"][0]["debt_stats"])
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 2 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        activations = []
        for epoch_data in data["epochs"]:
            stats = epoch_data["debt_stats"][layer_idx]
            activations.append(stats["total_activations"])

        activations = np.array(activations)
        num_experts = activations.shape[1]
        epochs = range(len(activations))

        for e in range(num_experts):
            ax.plot(epochs, activations[:, e], label=f"Expert {e}")

        ax.set_ylabel(f"Layer {layer_idx}")
        ax.legend(fontsize=7, ncol=num_experts)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Cumulative Expert Activations Over Training", fontsize=13)
    plt.tight_layout()

    out = save_path or str(FIGURE_DIR / "expert-activation-timeline.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_failure_analysis(log_dir: str = None, save_path: str = None):
    """When does GatedMoE fail? Accuracy gap vs learned routing as budget increases."""
    df = load_all_results(log_dir or "", prefix="sweep_")
    if df.empty:
        return

    from gatedmoe.exp_metric_collect import aggregate_by_method_budget
    agg = aggregate_by_method_budget(df)

    learned = agg[agg["method"] == "learned_top1"].sort_values("budget_kb")
    gatedmoe = agg[agg["method"] == "gatedmoe_g"].sort_values("budget_kb")

    if learned.empty or gatedmoe.empty:
        return

    # Merge on budget
    merged = pd.merge(
        learned[["budget_kb", "acc_mean"]].rename(columns={"acc_mean": "learned_acc"}),
        gatedmoe[["budget_kb", "acc_mean"]].rename(columns={"acc_mean": "gatedmoe_acc"}),
        on="budget_kb",
    )
    merged["gap"] = merged["learned_acc"] - merged["gatedmoe_acc"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(range(len(merged)), merged["gap"], color=["#2ecc71" if g <= 0 else "#e74c3c" for g in merged["gap"]])
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels([f"{int(b)}KB" for b in merged["budget_kb"]])
    ax.set_xlabel("SRAM Budget")
    ax.set_ylabel("Accuracy Gap (Learned - GatedMoE-G)")
    ax.set_title("When Does Learned Routing Win?")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out = save_path or str(FIGURE_DIR / "failure-analysis-gap.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
