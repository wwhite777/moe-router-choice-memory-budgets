"""E6-LM tests: gate/mask/logging semantics on the LM path."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter
from gatedmoe.lm_model_define import GatedMoELM, estimate_attention_cost, batchify
from gatedmoe.lm_train_loop import make_lm

WIDTHS = [64, 96, 128, 192, 256, 384, 512, 1024]
VOCAB = 500


def build(mask=None, method="gatedmoe_base", seed=0):
    cfg = dict(method=method, seed=seed, d_model=128, n_heads=4,
               d_ff_shared=256, widths=WIDTHS, num_layers=4, bptt=64,
               admissible=mask)
    return make_lm(cfg, VOCAB)


def run_step(model, budget, b=8, S=64):
    mc = VirtualMemoryCounter(budget)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(8) for _ in range(4)]
    x = torch.randint(0, VOCAB, (b, S))
    mc.reset_step()
    logits, info = model(x, gate, debt, mc)
    if not hasattr(model, "routers"):
        for i, dc in enumerate(debt):
            dc.step(info["activated_experts"][i])
    return mc, logits, info


def test_lm_forward_and_fsize():
    torch.manual_seed(0)
    model = build()
    mc = VirtualMemoryCounter(10 ** 12)
    gate = PredictiveMemoryGate(mc)
    bcost = model.backbone_cost(gate, 8, 64)
    layer = model.layers[0]
    costs = [gate.estimate_expert_cost(e, 8, 64) for e in layer.experts]
    # budget admitting exactly 3 experts
    budget = bcost + costs[2] + 1024
    mc2, logits, info = run_step(build(), budget)
    assert torch.isfinite(logits).all()
    assert info["feasible_sizes"] == [3] * 4, info["feasible_sizes"]
    assert mc2.peak_bytes <= budget


def test_lm_budget_zero_experts():
    torch.manual_seed(0)
    model = build()
    mc = VirtualMemoryCounter(10 ** 12)
    gate = PredictiveMemoryGate(mc)
    budget = model.backbone_cost(gate, 8, 64) + 1024  # backbone only
    mc2, _, info = run_step(build(), budget)
    assert info["feasible_sizes"] == [0] * 4
    assert all(a is None for a in info["activated_experts"])


def test_lm_masks_all_methods():
    mask = [1, 5]
    for method in ("gatedmoe_base", "round_robin", "random", "learned_top1"):
        torch.manual_seed(0)
        model = build(mask=mask, method=method, seed=3)
        _, _, info = run_step(model, 10 ** 10)
        assert info["feasible_sizes"] == [2] * 4, (method, info["feasible_sizes"])
        for a in info["activated_experts"]:
            assert a in mask, (method, a)


def test_attention_cost_positive_monotone():
    c1 = estimate_attention_cost(128, 4, 8, 64)
    c2 = estimate_attention_cost(128, 4, 16, 64)
    c3 = estimate_attention_cost(128, 4, 8, 128)
    assert 0 < c1 < c2 and c1 < c3


def test_batchify_shapes():
    data = torch.arange(1000)
    b = batchify(data, 16)
    assert b.shape == (16, 62)


if __name__ == "__main__":
    test_lm_forward_and_fsize()
    test_lm_budget_zero_experts()
    test_lm_masks_all_methods()
    test_attention_cost_positive_monotone()
    test_batchify_shapes()
    print("all LM tests passed")
