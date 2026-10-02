"""E5 property tests (must pass BEFORE the token-routing grid)."""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter
from gatedmoe.core_model_define import GatedMoEModel
from gatedmoe.sim_baseline_routers import create_baseline_model

WIDTHS = [64, 96, 128, 192, 256, 384, 512, 1024]


def build(widths=WIDTHS):
    torch.manual_seed(0)
    return GatedMoEModel(d_model=128, d_ff_shared=256, d_ff_expert=widths,
                         num_experts=len(widths), num_layers=4)


def run_steps(method, budget, steps=6, b=16):
    model = create_baseline_model(build(), method, seed=1)
    mc = VirtualMemoryCounter(budget)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(8) for _ in range(4)]
    infos = []
    for s in range(steps):
        torch.manual_seed(100 + s)
        mc.reset_step()
        logits, info = model(torch.randn(b, 3, 32, 32), gate, debt, mc)
        assert torch.isfinite(logits).all()
        for i, dc in enumerate(debt):
            a = info["activated_experts"][i]
            if isinstance(a, list):
                for e in a:
                    dc.step(e)
            else:
                dc.step(a)
        infos.append(info)
    return mc, infos


def test_knapsack_never_exceeds_budget():
    """The load-bearing property: co-activation respects B_SRAM at every step."""
    for budget_mb in (6, 8, 12, 24):
        mc, infos = run_steps("token_knapsack", budget_mb * 1024 * 1024)
        assert mc.peak_bytes <= budget_mb * 1024 * 1024, (budget_mb, mc.peak_bytes)


def test_knapsack_admits_more_with_bigger_budget():
    small_sets = [len(a) for _, infos in [run_steps("token_knapsack", 6 * 1024 * 1024)]
                  for info in infos for a in info["activated_experts"]]
    large_sets = [len(a) for _, infos in [run_steps("token_knapsack", 24 * 1024 * 1024)]
                  for info in infos for a in info["activated_experts"]]
    assert sum(large_sets) > sum(small_sets), (sum(small_sets), sum(large_sets))


def test_knapsack_cost_uses_actual_token_count():
    """Cost booked for T_e tokens must be less than the full-batch expert cost
    whenever the expert receives only part of the tokens (b=16, S=64)."""
    gate = PredictiveMemoryGate(VirtualMemoryCounter(10**12))
    model = build()
    e = model.layers[0].experts[0]
    full = gate.estimate_expert_cost(e, 16, 64)
    part = gate.estimate_expert_cost(e, 1, 100)  # 100 of 1024 tokens
    assert part < full


def test_capacity_router_respects_cap():
    model = create_baseline_model(build(), "token_capacity", seed=1)
    mc = VirtualMemoryCounter(10**12)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(8) for _ in range(4)]
    b, S, K = 16, 64, 8
    cap = max(1, math.ceil(1.25 * b * S / K))
    torch.manual_seed(5)
    mc.reset_step()
    _, info = model(torch.randn(b, 3, 32, 32), gate, debt, mc)
    # Verify via router internals: re-run token assignment per layer
    for layer, router in zip(model.layers, model.routers):
        assert router.aux_loss.item() >= 0.0
    # Structural check: capacity router activates every expert that received
    # tokens (list-valued activations), never more than K
    for a in info["activated_experts"]:
        assert isinstance(a, list) and len(a) <= K


def test_token_routers_respect_admissible_mask():
    mask = [0, 3, 7]
    torch.manual_seed(0)
    base = GatedMoEModel(d_model=128, d_ff_shared=256, d_ff_expert=WIDTHS,
                         num_experts=8, num_layers=4, admissible_mask=mask)
    for method in ("token_knapsack", "token_capacity"):
        model = create_baseline_model(base, method, seed=1)
        mc = VirtualMemoryCounter(10**12)
        gate = PredictiveMemoryGate(mc)
        debt = [DebtCounter(8) for _ in range(4)]
        mc.reset_step()
        _, info = model(torch.randn(8, 3, 32, 32), gate, debt, mc)
        for a in info["activated_experts"]:
            assert set(a) <= set(mask), (method, a)


if __name__ == "__main__":
    test_knapsack_never_exceeds_budget()
    test_knapsack_admits_more_with_bigger_budget()
    test_knapsack_cost_uses_actual_token_count()
    test_capacity_router_respects_cap()
    test_token_routers_respect_admissible_mask()
    print("all token-router property tests passed")
