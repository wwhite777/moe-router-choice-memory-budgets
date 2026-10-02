"""Design the E1/E4 budget-sweep grids from the analytical gate itself.

Computes shared + per-expert Chat costs with the real PredictiveMemoryGate,
derives the budget thresholds where each expert becomes feasible, picks one
budget per feasible-set size |F| = 0..K, and writes configs/e1_grid.json and
configs/e4_grid.json job lists for the queue runner.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_model_define import SharedBase, Expert

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

E1 = dict(
    tag="e1", d_model=128, d_ff_shared=256, patch_size=4,
    widths=[64, 96, 128, 192, 256, 384, 512, 1024],
    batch_size=16, datasets=["cifar10", "cifar100"],
    seeds=[42, 123, 7, 2026, 31415], epochs=30,
)
E4 = dict(
    tag="e4", d_model=32, d_ff_shared=64, patch_size=8,
    widths=[32, 48, 64, 96, 128, 192, 256, 512],
    batch_size=8, datasets=["cifar10"],
    seeds=[42, 123, 7, 2026, 31415], epochs=30,
    # Named-device anchors: STM32F407 192KB, MCUNet-class 256KB, STM32F746
    # 320KB, ESP32-S3 512KB, STM32H743 1MB, i.MX RT1170 2MB.
    device_budgets=[192 * 1024, 256 * 1024, 320 * 1024, 512 * 1024,
                    1024 * 1024, 2 * 1024 * 1024],
)
METHODS = ["round_robin", "gatedmoe_base", "gatedmoe_g", "learned_top1", "random"]

# S1 controlled choice-set study (PROTOCOL.md §3): equal-cost experts, fixed
# 8 MB budget, admissible mask is the only constraint.
S1 = dict(
    tag="s1", d_model=128, d_ff_shared=256, patch_size=4,
    widths=[256] * 8, batch_size=16, datasets=["cifar100"],
    seeds=[42, 123, 7, 2026, 31415], epochs=30,
    budget=8 * 1024 * 1024,
    mask_sizes=[1, 2, 3, 4, 6, 8], subsets_per_size=2, mask_rng_seed=777,
)
S1_METHODS = ["round_robin", "gatedmoe_base", "learned_top1", "random"]


def s1_masks(spec):
    """Frozen admissible subsets: 2 per size (1 for the full set), rng seed 777."""
    import random as _random
    rng = _random.Random(spec["mask_rng_seed"])
    K = len(spec["widths"])
    cells = []
    for m in spec["mask_sizes"]:
        n_sub = 1 if m == K else spec["subsets_per_size"]
        seen = set()
        for v in range(n_sub):
            while True:
                subset = tuple(sorted(rng.sample(range(K), m)))
                if subset not in seen:
                    seen.add(subset)
                    break
            cells.append({"m": m, "variant": v, "mask": list(subset)})
    return cells


def make_s1_jobs(spec):
    cells = s1_masks(spec)
    jobs = []
    for seed_rank, seed in enumerate(spec["seeds"]):
        for dataset in spec["datasets"]:
            for cell in cells:
                for method in S1_METHODS:
                    ds = "c10" if dataset == "cifar10" else "c100"
                    name = (f"{spec['tag']}_{ds}_{method}_m{cell['m']}"
                            f"v{cell['variant']}_s{seed}")
                    jobs.append({
                        "name": name, "method": method, "dataset": dataset,
                        "budget": spec["budget"], "seed": seed, "seed_rank": seed_rank,
                        "widths": spec["widths"], "d_model": spec["d_model"],
                        "d_ff_shared": spec["d_ff_shared"], "patch_size": spec["patch_size"],
                        "batch_size": spec["batch_size"], "epochs": spec["epochs"],
                        "admissible": cell["mask"],
                        "expected_fsize": cell["m"],
                    })
    return jobs


def cost_table(spec):
    mc = VirtualMemoryCounter(10**12)
    gate = PredictiveMemoryGate(mc)
    b = spec["batch_size"]
    S = (32 // spec["patch_size"]) ** 2
    spec["seq_len"] = S
    modules = [Expert(spec["d_model"], w) for w in spec["widths"]]  # keep refs alive
    shared = gate.estimate_shared_cost(SharedBase(spec["d_model"], spec["d_ff_shared"]), b, S)
    expert_costs = [gate.estimate_expert_cost(m, b, S) for m in modules]
    thresholds = sorted(shared + c for c in expert_costs)
    return shared, expert_costs, thresholds


def fsize_at(budget, shared, expert_costs):
    return sum(1 for c in expert_costs if shared + c <= budget)


def pick_budgets(shared, thresholds):
    """One budget per |F| = 0..K: midpoints between consecutive thresholds."""
    budgets = []
    lo = shared + 64 * 1024  # |F| = 0 point: shared fits, nothing else
    budgets.append(min(lo, thresholds[0] - 64 * 1024))
    for i in range(len(thresholds)):
        hi = thresholds[i + 1] if i + 1 < len(thresholds) else thresholds[-1] * 1.15
        budgets.append(int((thresholds[i] + hi) / 2))
    return [int(b) for b in budgets]


def make_jobs(spec, budgets, shared, expert_costs):
    jobs = []
    for seed_rank, seed in enumerate(spec["seeds"]):
        for dataset in spec["datasets"]:
            for budget in budgets:
                for method in METHODS:
                    ds = "c10" if dataset == "cifar10" else "c100"
                    name = f"{spec['tag']}_{ds}_{method}_B{budget // 1024}k_s{seed}"
                    jobs.append({
                        "name": name, "method": method, "dataset": dataset,
                        "budget": budget, "seed": seed, "seed_rank": seed_rank,
                        "widths": spec["widths"], "d_model": spec["d_model"],
                        "d_ff_shared": spec["d_ff_shared"], "patch_size": spec["patch_size"],
                        "batch_size": spec["batch_size"], "epochs": spec["epochs"],
                        "expected_fsize": fsize_at(budget, shared, expert_costs),
                    })
    return jobs


def report(spec, budgets):
    shared, expert_costs, thresholds = cost_table(spec)
    print(f"\n== {spec['tag'].upper()}  d={spec['d_model']} f_s={spec['d_ff_shared']} "
          f"b={spec['batch_size']} S={spec['seq_len']} ==")
    print(f"shared cost: {shared / 1024:.0f} KB")
    for w, c in zip(spec["widths"], expert_costs):
        print(f"  f={w:5d}  Chat={c / 1024:8.0f} KB  threshold(shared+Chat)={(shared + c) / 1024:8.0f} KB")
    for b in budgets:
        print(f"  budget {b / 1024:8.0f} KB -> |F| = {fsize_at(b, shared, expert_costs)}")
    return shared, expert_costs


# E5 token-routing arm: 3 budgets spanning small/medium/full
# feasibility, arms = knapsack vs capacity-factor control vs batch-level
# learned top-1. Capacity factor default 1.25; tune-once runs happen at the
# largest budget per plan before confirmatory seeds.
E5_BUDGETS = [6248448, 8591360, 17766400]  # |F|=2 / |F|=5 (8MB-class) / |F|=8
E5_METHODS = ["token_knapsack", "token_capacity", "learned_top1"]


def make_e5_jobs(spec, budgets):
    jobs = []
    for seed_rank, seed in enumerate(spec["seeds"]):
        for dataset in spec["datasets"]:
            for budget in budgets:
                for method in E5_METHODS:
                    ds = "c10" if dataset == "cifar10" else "c100"
                    name = f"e5_{ds}_{method}_B{budget // 1024}k_s{seed}"
                    jobs.append({
                        "name": name, "method": method, "dataset": dataset,
                        "budget": budget, "seed": seed, "seed_rank": seed_rank,
                        "widths": spec["widths"], "d_model": spec["d_model"],
                        "d_ff_shared": spec["d_ff_shared"], "patch_size": spec["patch_size"],
                        "batch_size": spec["batch_size"], "epochs": spec["epochs"],
                    })
    return jobs


def main():
    # S1: verify every expert individually fits at the fixed budget
    s_shared, s_costs, _ = cost_table(S1)
    assert all(s_shared + c <= S1["budget"] for c in s_costs), \
        "S1 invariant violated: some equal-cost expert does not fit at 8 MB"
    s1_jobs = make_s1_jobs(S1)
    cells = s1_masks(S1)
    with open(CONFIG_DIR / "s1_grid.json", "w") as f:
        json.dump({"budget": S1["budget"], "mask_cells": cells, "jobs": s1_jobs}, f, indent=1)
    print(f"S1: shared={s_shared // 1024}KB expert={s_costs[0] // 1024}KB "
          f"(x8 equal) budget={S1['budget'] // 1024}KB")
    for c in cells:
        print(f"  m={c['m']} v{c['variant']}: {c['mask']}")
    print(f"S1 jobs: {len(s1_jobs)}")

    shared, expert_costs, thresholds = cost_table(E1)
    e1_budgets = pick_budgets(shared, thresholds)
    report(E1, e1_budgets)
    e1_jobs = make_jobs(E1, e1_budgets, shared, expert_costs)
    with open(CONFIG_DIR / "e1_grid.json", "w") as f:
        json.dump({"budgets": e1_budgets, "jobs": e1_jobs}, f, indent=1)
    print(f"E1 jobs: {len(e1_jobs)}")

    shared4, expert_costs4, _ = cost_table(E4)
    e4_budgets = E4["device_budgets"]
    report(E4, e4_budgets)
    e4_jobs = make_jobs(E4, e4_budgets, shared4, expert_costs4)
    with open(CONFIG_DIR / "e4_grid.json", "w") as f:
        json.dump({"budgets": e4_budgets, "jobs": e4_jobs}, f, indent=1)
    print(f"E4 jobs: {len(e4_jobs)}")

    e5_jobs = make_e5_jobs(E1, E5_BUDGETS)
    with open(CONFIG_DIR / "e5_grid.json", "w") as f:
        json.dump({"budgets": E5_BUDGETS, "jobs": e5_jobs}, f, indent=1)
    print(f"E5 jobs: {len(e5_jobs)} (launch AFTER property tests + G-SM1 slot check)")


if __name__ == "__main__":
    main()
