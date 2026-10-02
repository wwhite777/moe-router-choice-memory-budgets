"""Tests for feasible-set logging and the content-keyed cost cache."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter
from gatedmoe.core_model_define import GatedMoEModel, Expert
from gatedmoe.sim_baseline_routers import create_baseline_model

WIDTHS = [64, 96, 128, 192, 256, 384, 512, 1024]


def build(num_classes=10):
    return GatedMoEModel(
        d_model=128, d_ff_shared=256, d_ff_expert=WIDTHS,
        num_experts=8, num_layers=4, num_classes=num_classes,
    )


def analytic_fsize(model, gate, budget, b=16, S=64):
    layer = model.layers[0]
    shared = gate.estimate_shared_cost(layer.shared, b, S)
    return sum(
        1 for e in layer.experts
        if shared + gate.estimate_expert_cost(e, b, S) <= budget
    )


def test_cost_cache_content_keyed():
    """Temporary modules with reused ids must not collide in the cache."""
    gate = PredictiveMemoryGate(VirtualMemoryCounter(10**12))
    costs = [gate.estimate_expert_cost(Expert(128, w), 16, 64) for w in WIDTHS]
    assert len(set(costs)) == len(WIDTHS), f"cache collision: {costs}"
    assert costs == sorted(costs), "cost must be monotone in width"


def test_fsize_logged_gatedmoe_matches_analytic():
    torch.manual_seed(0)
    model = build()
    for budget_kb, expected in [(3907, 0), (6629, 3), (8390, 5), (17350, 8)]:
        mc = VirtualMemoryCounter(budget_kb * 1024)
        gate = PredictiveMemoryGate(mc)
        debt = [DebtCounter(8) for _ in range(4)]
        x = torch.randn(16, 3, 32, 32)
        mc.reset_step()
        _, info = model(x, gate, debt, mc)
        assert info["feasible_sizes"] == [expected] * 4, (
            budget_kb, expected, info["feasible_sizes"])
        assert analytic_fsize(model, gate, budget_kb * 1024) == expected


def test_fsize_logged_baselines_match_gatedmoe():
    torch.manual_seed(0)
    base = build()
    for method in ["round_robin", "random", "learned_top1"]:
        model = create_baseline_model(build(), method)
        for budget_kb, expected in [(3907, 0), (7334, 4), (17350, 8)]:
            mc = VirtualMemoryCounter(budget_kb * 1024)
            gate = PredictiveMemoryGate(mc)
            debt = [DebtCounter(8) for _ in range(4)]
            x = torch.randn(16, 3, 32, 32)
            mc.reset_step()
            _, info = model(x, gate, debt, mc)
            assert info["feasible_sizes"] == [expected] * 4, (
                method, budget_kb, expected, info["feasible_sizes"])


def test_budget_respected_when_expert_active():
    """Per-step peak must never exceed budget for feasibility-aware methods."""
    torch.manual_seed(0)
    model = build()
    mc = VirtualMemoryCounter(8390 * 1024)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(8) for _ in range(4)]
    for _ in range(10):
        mc.reset_step()
        _, info = model(torch.randn(16, 3, 32, 32), gate, debt, mc)
        for i, dc in enumerate(debt):
            dc.step(info["activated_experts"][i])
    assert mc.peak_bytes <= 8390 * 1024, (mc.peak_bytes, 8390 * 1024)


def test_grid_files_realize_all_fsizes():
    """E1.0 pre-launch gate: budget grids must realize the
    claimed |F| ladder, monotone in budget, for both backbones; S1 masks must
    equal their declared m."""
    import json
    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    e1 = json.load(open(cfg_dir / "e1_grid.json"))
    fs = {}
    for j in e1["jobs"]:
        fs[j["budget"]] = j["expected_fsize"]
    sizes = [fs[b] for b in sorted(fs)]
    assert sizes == list(range(9)), f"E1 grid must realize |F|=0..8, got {sizes}"

    e4 = json.load(open(cfg_dir / "e4_grid.json"))
    fs4 = {}
    for j in e4["jobs"]:
        fs4[j["budget"]] = j["expected_fsize"]
    sizes4 = [fs4[b] for b in sorted(fs4)]
    assert sizes4 == sorted(sizes4), f"E4 |F| must be monotone in budget: {sizes4}"
    assert sizes4[0] == 0 and sizes4[-1] == 8, f"E4 must span 0..8: {sizes4}"

    s1 = json.load(open(cfg_dir / "s1_grid.json"))
    for j in s1["jobs"]:
        assert len(j["admissible"]) == j["expected_fsize"]
        assert len(set(j["admissible"])) == len(j["admissible"])


def test_admissible_mask_restricts_all_methods():
    """With an ample budget, |F| must equal the mask size and every activation
    must come from the mask — for the debt model and every baseline router."""
    mask = [1, 4, 6]
    budget = 32 * 1024 * 1024

    def build_masked():
        torch.manual_seed(0)
        return GatedMoEModel(
            d_model=128, d_ff_shared=256, d_ff_expert=[256] * 8,
            num_experts=8, num_layers=4, admissible_mask=mask,
        )

    for method in [None, "round_robin", "random", "learned_top1"]:
        model = build_masked() if method is None else create_baseline_model(
            build_masked(), method, seed=42)
        mc = VirtualMemoryCounter(budget)
        gate = PredictiveMemoryGate(mc)
        debt = [DebtCounter(8) for _ in range(4)]
        for _ in range(8):
            mc.reset_step()
            _, info = model(torch.randn(4, 3, 32, 32), gate, debt, mc)
            assert info["feasible_sizes"] == [3] * 4, (method, info["feasible_sizes"])
            for a in info["activated_experts"]:
                assert a in mask, (method, a)
            for i, dc in enumerate(debt):
                dc.step(info["activated_experts"][i])


def test_random_router_rng_isolated():
    """RandomRouter must never consume the global torch RNG stream."""
    model = create_baseline_model(build(), "random", seed=7)
    mc = VirtualMemoryCounter(32 * 1024 * 1024)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(8) for _ in range(4)]
    x = torch.randn(4, 3, 32, 32)
    state_before = torch.random.get_rng_state().clone()
    mc.reset_step()
    model(x, gate, debt, mc)
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after), "global RNG was consumed"


if __name__ == "__main__":
    test_cost_cache_content_keyed()
    test_fsize_logged_gatedmoe_matches_analytic()
    test_fsize_logged_baselines_match_gatedmoe()
    test_budget_respected_when_expert_active()
    test_grid_files_realize_all_fsizes()
    test_admissible_mask_restricts_all_methods()
    test_random_router_rng_isolated()
    print("all feasible-logging tests passed")
