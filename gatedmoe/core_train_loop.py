"""Training loop for GatedMoE and all baselines.

Handles forward/backward, debt updates, memory tracking, and metric logging.
"""

import copy
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from typing import Optional
from pathlib import Path

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter, GradientDebtCounter
from gatedmoe.core_model_define import GatedMoEModel, build_model_from_config
from gatedmoe.sim_memory_profile import profile_model_step, StepMetrics
from gatedmoe.sim_baseline_routers import create_baseline_model


# ─── Data ────────────────────────────────────────────────────────────────────

def get_cifar_loaders(batch_size: int, num_workers: int = 4, data_dir: str = "./data",
                      dataset: str = "cifar10", seed: int = 0, val_size: int = 5000):
    """Train (45k) / val (5k, fixed last indices, test transforms) / test loaders.

    PROTOCOL.md §1: validation indices are seed-independent; the train loader
    uses a dedicated generator so batch ORDER is identical across methods at
    the same seed (matched comparisons).
    """
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    DS = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10
    train_full = DS(root=data_dir, train=True, download=True, transform=transform_train)
    val_full = DS(root=data_dir, train=True, download=True, transform=transform_test)
    test_ds = DS(root=data_dir, train=False, download=True, transform=transform_test)
    n = len(train_full)
    train_ds = torch.utils.data.Subset(train_full, range(0, n - val_size))
    val_ds = torch.utils.data.Subset(val_full, range(n - val_size, n))

    g = torch.Generator()
    g.manual_seed(1000 + seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─── Training ────────────────────────────────────────────────────────────────

def resolve_step_budget(cfg, step_idx: int) -> int:
    schedule = getattr(cfg.memory, "budget_schedule", "static")
    max_budget = cfg.memory.budget_bytes
    min_budget = getattr(cfg.memory, "budget_min_bytes", max_budget)
    cycle_steps = max(1, getattr(cfg.memory, "budget_cycle_steps", 1))

    if schedule == "static" or min_budget >= max_budget:
        return max_budget
    if schedule == "alternate":
        block_idx = (step_idx // cycle_steps) % 2
        return min_budget if block_idx == 0 else max_budget

    raise ValueError(f"Unknown budget schedule: {schedule}")


def apply_step_budget(mc: VirtualMemoryCounter, cfg, step_idx: int) -> int:
    budget = resolve_step_budget(cfg, step_idx)
    mc.budget = budget
    return budget


def train_one_epoch(
    model: GatedMoEModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gate: PredictiveMemoryGate,
    debt_counters: list,
    mc: VirtualMemoryCounter,
    device: torch.device,
    cfg,
    global_step: int = 0,
    variant: str = "gatedmoe_g",
    route_trace: Optional[list] = None,
    route_class_counts=None,  # np array [L, K+1, C]; slot 0 = null path
) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_macs = 0
    step_count = 0
    budget_min = None
    budget_max = None
    num_experts = cfg.model.num_experts
    feasible_hist = [0] * (num_experts + 1)  # counts over (step, layer) events

    for images, labels in loader:
        step_budget = apply_step_budget(mc, cfg, global_step)
        budget_min = step_budget if budget_min is None else min(budget_min, step_budget)
        budget_max = step_budget if budget_max is None else max(budget_max, step_budget)
        images, labels = images.to(device), labels.to(device)
        mc.reset_step()

        logits, info = model(images, gate, debt_counters, mc)
        loss = F.cross_entropy(logits, labels)

        # Add auxiliary load-balancing loss for learned routers
        if hasattr(model, 'routers'):
            for r in model.routers:
                if hasattr(r, 'aux_loss') and isinstance(r.aux_loss, torch.Tensor):
                    loss = loss + 0.01 * r.aux_loss

        optimizer.zero_grad()
        loss.backward()

        # Update debt counters
        current_loss_val = loss.item()
        for i, dc in enumerate(debt_counters):
            activated = info["activated_experts"][i]
            # Dense returns list of all experts; normalize to single int or None
            if isinstance(activated, list):
                activated = activated[0] if activated else None
            if isinstance(dc, GradientDebtCounter):
                dc.step(activated, current_loss=current_loss_val)
            else:
                dc.step(activated)

        optimizer.step()

        for fsize in info.get("feasible_sizes", []):
            feasible_hist[min(fsize, num_experts)] += 1

        # Route diagnostics (PROTOCOL.md §6)
        step_routes = []
        for a in info["activated_experts"]:
            step_routes.append(-2 if isinstance(a, list) else (-1 if a is None else a))
        if route_trace is not None:
            route_trace.append(step_routes)
        if route_class_counts is not None:
            class_hist = torch.bincount(
                labels, minlength=route_class_counts.shape[2]).cpu().numpy()
            for l, a in enumerate(info["activated_experts"]):
                acts = a if isinstance(a, list) else ([] if a is None else [a])
                if not acts:
                    route_class_counts[l, 0] += class_hist
                for e in acts:
                    route_class_counts[l, e + 1] += class_hist

        # Metrics
        metrics = profile_model_step(model, info["activated_experts"], images.shape[0], mc)
        total_loss += loss.item() * images.shape[0]
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.shape[0]
        total_macs += metrics.macs
        step_count += 1
        global_step += 1

    total_events = max(1, sum(feasible_hist))
    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
        "avg_macs_per_step": total_macs / step_count,
        "peak_sram_bytes": mc.peak_bytes,
        "flash_to_sram_bytes": mc.flash_to_sram_bytes,
        "budget_bytes_min": budget_min,
        "budget_bytes_max": budget_max,
        "feasible_hist": feasible_hist,
        "mean_feasible": sum(i * c for i, c in enumerate(feasible_hist)) / total_events,
        "frac_gate_constrained": sum(
            c for i, c in enumerate(feasible_hist) if i < num_experts
        ) / total_events,
        "frac_routing_choice": sum(
            c for i, c in enumerate(feasible_hist) if i >= 2
        ) / total_events,
    }, global_step


@torch.no_grad()
def evaluate(
    model: GatedMoEModel,
    loader: DataLoader,
    gate: PredictiveMemoryGate,
    debt_counters: list,
    mc: VirtualMemoryCounter,
    device: torch.device,
    cfg,
) -> dict:
    model.eval()
    eval_counters = copy.deepcopy(debt_counters)
    router_positions = None
    router_gen_states = None
    if hasattr(model, "routers"):
        router_positions = [getattr(router, "current", None) for router in model.routers]
        # Isolated-RNG routers: snapshot generator state so eval draws never
        # perturb the training-time stream.
        router_gen_states = [
            router.gen.get_state().clone() if hasattr(router, "gen") else None
            for router in model.routers
        ]
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    step_idx = 0

    try:
        for images, labels in loader:
            apply_step_budget(mc, cfg, step_idx)
            images, labels = images.to(device), labels.to(device)
            mc.reset_step()
            logits, info = model(images, gate, eval_counters, mc)
            loss = F.cross_entropy(logits, labels)

            if not hasattr(model, "routers"):
                for i, dc in enumerate(eval_counters):
                    activated = info["activated_experts"][i]
                    if isinstance(activated, list):
                        activated = activated[0] if activated else None
                    dc.step(activated)

            total_loss += loss.item() * images.shape[0]
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.shape[0]
            step_idx += 1
    finally:
        if router_positions is not None:
            for router, current in zip(model.routers, router_positions):
                if current is not None:
                    router.current = current
        if router_gen_states is not None:
            for router, state in zip(model.routers, router_gen_states):
                if state is not None:
                    router.gen.set_state(state)

    return {
        "test_loss": total_loss / total_samples,
        "test_acc": total_correct / total_samples,
    }


# ─── Full run ────────────────────────────────────────────────────────────────

def run_experiment(cfg) -> dict:
    """Run a full training experiment from config."""
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # Dataset must be resolved BEFORE the model is built (num_classes).
    dataset = getattr(cfg, 'dataset', 'cifar10')
    if dataset == "cifar100":
        cfg.model.num_classes = 100

    # Model
    base_model = build_model_from_config(cfg).to(device)

    # Memory infrastructure
    mc = VirtualMemoryCounter(cfg.memory.budget_bytes, cfg.memory.dtype_bytes)
    gate = PredictiveMemoryGate(mc, cfg.memory.dtype_bytes)

    # Debt counters (one per layer)
    debt_counters = []
    for _ in range(cfg.model.num_layers):
        if cfg.method == "gatedmoe_g":
            debt_counters.append(GradientDebtCounter(cfg.model.num_experts))
        elif cfg.method == "gatedmoe_base":
            debt_counters.append(DebtCounter(cfg.model.num_experts))
        else:
            debt_counters.append(DebtCounter(cfg.model.num_experts))

    # Wrap with baseline router if needed
    baseline_methods = {"dense", "learned_top1", "random", "round_robin", "mod",
                        "token_knapsack", "token_capacity", "gated_top1"}
    if cfg.method in baseline_methods:
        model = create_baseline_model(
            base_model, cfg.method, seed=cfg.train.seed,
            capacity_factor=getattr(cfg.model, "capacity_factor", 1.25)).to(device)
    else:
        model = base_model

    # Data (train 45k / val 5k fixed split / test; matched batch order per seed)
    train_loader, val_loader, test_loader = get_cifar_loaders(
        cfg.train.batch_size, cfg.train.num_workers, dataset=dataset,
        seed=cfg.train.seed)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # Training
    results = {"epochs": [], "config": {
        "method": cfg.method,
        "dataset": dataset,
        "budget_bytes": cfg.memory.budget_bytes,
        "budget_schedule": getattr(cfg.memory, "budget_schedule", "static"),
        "budget_min_bytes": getattr(cfg.memory, "budget_min_bytes", cfg.memory.budget_bytes),
        "budget_cycle_steps": getattr(cfg.memory, "budget_cycle_steps", 1),
        "batch_size": cfg.train.batch_size,
        "num_experts": cfg.model.num_experts,
        "seed": cfg.train.seed,
        "d_model": cfg.model.d_model,
        "d_ff_shared": cfg.model.d_ff_shared,
        "d_ff_expert": cfg.model.d_ff_expert,
        "epochs": cfg.train.epochs,
        "admissible_mask": getattr(cfg.model, "admissible_mask", None),
        "protocol": "s0",
    }}

    import numpy as np
    route_trace = []
    route_class_counts = np.zeros(
        (cfg.model.num_layers, cfg.model.num_experts + 1, cfg.model.num_classes),
        dtype=np.int64)

    best_acc = 0.0
    best_val_acc = -1.0
    best_val_epoch = -1
    test_acc_at_best_val = 0.0
    global_step = 0
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, gate, debt_counters, mc, device, cfg,
            global_step=global_step, variant=cfg.method,
            route_trace=route_trace, route_class_counts=route_class_counts,
        )
        val_metrics = evaluate(model, val_loader, gate, debt_counters, mc, device, cfg)
        test_metrics = evaluate(model, test_loader, gate, debt_counters, mc, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0

        epoch_data = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["test_loss"],
            "val_acc": val_metrics["test_acc"],
            **test_metrics,
            "time_s": elapsed,
            "debt_stats": [dc.get_stats() for dc in debt_counters],
        }
        results["epochs"].append(epoch_data)

        if val_metrics["test_acc"] > best_val_acc:
            best_val_acc = val_metrics["test_acc"]
            best_val_epoch = epoch
            test_acc_at_best_val = test_metrics["test_acc"]
        if test_metrics["test_acc"] > best_acc:
            best_acc = test_metrics["test_acc"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{cfg.experiment_name}] Epoch {epoch+1:3d} | "
                f"Train {train_metrics['train_acc']:.4f} | "
                f"Test {test_metrics['test_acc']:.4f} | "
                f"Peak SRAM {train_metrics['peak_sram_bytes']/1024:.1f}KB | "
                f"Budget {train_metrics['budget_bytes_min']/1024:.0f}-"
                f"{train_metrics['budget_bytes_max']/1024:.0f}KB | "
                f"{elapsed:.1f}s"
            )

    results["best_test_acc"] = best_acc  # continuity only; BANNED from manuscript
    results["final_test_acc"] = results["epochs"][-1]["test_acc"]  # PRIMARY endpoint
    results["final_test_loss"] = results["epochs"][-1]["test_loss"]
    results["best_val_epoch"] = best_val_epoch
    results["best_val_acc"] = best_val_acc
    results["test_acc_at_best_val"] = test_acc_at_best_val  # SECONDARY endpoint

    # Save
    save_dir = Path(cfg.save_dir) / cfg.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "routes_train.npy",
            np.asarray(route_trace, dtype=np.int8))
    np.save(save_dir / "route_class_counts.npy", route_class_counts)
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
