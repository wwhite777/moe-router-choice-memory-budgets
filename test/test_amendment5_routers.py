"""Amendment 5 (2026-10-01) tests: the task-trained GatedTop1Router and the dense ceiling.

1. Task gradient: with the TASK loss only (no balancing term), the gate of GatedTop1Router
   receives a non-zero gradient; the gate of LearnedTop1Router receives none (this pair
   also shows that the check can fail).
2. Forward identity: GatedTop1Router returns shared_out + p_chosen * expert_out, and the
   chosen expert is the highest-probability admissible expert that fits the gate.
3. Feasibility: under a budget that admits only some experts, and with an admissible
   mask, the gated router never selects an infeasible or inadmissible expert, even when
   its logits prefer one.
4. Vision smoke: a tiny model with gated_top1 and with dense runs one training step and
   one evaluation (core_train_loop.train_one_epoch / evaluate) without error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import torch
import torch.nn.functional as F

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter
from gatedmoe.core_model_define import GatedMoEModel, GatedMoELayer
from gatedmoe.sim_baseline_routers import (LearnedTop1Router, GatedTop1Router,
                                           create_baseline_model)
from gatedmoe.util_config_io import ExperimentConfig

D, WIDTHS = 16, [8, 12, 16, 24, 32, 48, 64, 128]
K = len(WIDTHS)
B, S = 4, 16


def _layer(mask=None):
    return GatedMoELayer(D, 32, WIDTHS, K, admissible_mask=mask)


def _model(method, mask=None):
    torch.manual_seed(0)
    base = GatedMoEModel(d_model=D, d_ff_shared=32, d_ff_expert=WIDTHS, num_experts=K,
                         num_layers=2, num_classes=10, image_size=32, patch_size=8,
                         admissible_mask=mask)
    return create_baseline_model(base, method, seed=7)


def _expected_choice(probs, layer, gate, mc):
    """Highest-probability admissible expert that fits (reference implementation)."""
    order = sorted(range(K), key=lambda i: -float(probs[i].detach()))
    for i in order:
        if layer.admissible[i] and mc.can_fit(gate.estimate_expert_cost(layer.experts[i], B, S)):
            return i
    return None


def _task_grad(method):
    model = _model(method)
    mc = VirtualMemoryCounter(10 ** 12)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(K) for _ in range(2)]
    torch.manual_seed(1)
    x = torch.randn(B, 3, 32, 32)
    y = torch.randint(0, 10, (B,))
    mc.reset_step()
    logits, info = model(x, gate, debt, mc)
    assert all(a is not None for a in info["activated_experts"])
    F.cross_entropy(logits, y).backward()  # TASK loss only, no aux term
    return [r.gate_proj.weight.grad for r in model.routers]


def test_task_gradient_reaches_gated_router_only():
    gated = _task_grad("gated_top1")
    learned = _task_grad("learned_top1")
    for g in gated:
        assert g is not None and g.abs().sum().item() > 0, "gated router got no task gradient"
    for g in learned:
        assert g is None or g.abs().sum().item() == 0, "learned router got a task gradient"


def test_aux_loss_identical_definition():
    torch.manual_seed(3)
    lr, gr = LearnedTop1Router(D, K), GatedTop1Router(D, K)
    gr.load_state_dict(lr.state_dict())
    h = torch.randn(B, S, D)
    pl, pg = lr(h), gr(h)
    assert torch.equal(pl, pg)
    assert torch.equal(lr.aux_loss, gr.aux_loss)
    assert torch.allclose(gr.aux_loss, (pg * pg).sum())


def test_forward_identity_and_choice():
    for seed in range(20):
        torch.manual_seed(seed)
        layer = _layer()
        router = GatedTop1Router(D, K)
        h = torch.randn(B, S, D)
        mc = VirtualMemoryCounter(10 ** 12)
        gate = PredictiveMemoryGate(mc)
        mc.reset_step()
        out, chosen = router.route(layer, h, gate, mc, B, S)
        probs = router(h, admissible=layer.admissible)
        ref_mc = VirtualMemoryCounter(10 ** 12)
        expect = _expected_choice(probs, layer, PredictiveMemoryGate(ref_mc), ref_mc)
        assert chosen == expect == int(torch.argmax(probs))
        ref = layer.shared(h) + probs[chosen] * layer.experts[chosen](h)
        assert torch.allclose(out, ref, atol=1e-6)
        # and it differs from the unweighted (learned_top1) output
        unweighted = layer.shared(h) + layer.experts[chosen](h)
        assert not torch.allclose(out, unweighted, atol=1e-6)


def test_feasibility_and_admissibility():
    mask = [0, 1, 3, 5, 6, 7]                     # experts 2 and 4 inadmissible
    gate0 = PredictiveMemoryGate(VirtualMemoryCounter(10 ** 12))
    layer0 = _layer(mask)
    costs = [gate0.estimate_expert_cost(e, B, S) for e in layer0.experts]
    budget = costs[4] + 1                          # experts 0..4 fit, 5..7 do not
    feasible_ok = {i for i in range(K) if costs[i] <= budget and i in mask}
    assert feasible_ok == {0, 1, 3}
    seen = set()
    for seed in range(60):
        torch.manual_seed(seed)
        layer = _layer(mask)
        router = GatedTop1Router(D, K)
        with torch.no_grad():                      # force a preference for a forbidden expert
            target = [2, 4, 5, 6, 7][seed % 5]
            router.gate_proj.weight.mul_(0.01)
            router.gate_proj.weight[target] += 5.0 * torch.sign(torch.randn(D))
        h = torch.randn(B, S, D)
        mc = VirtualMemoryCounter(budget)
        gate = PredictiveMemoryGate(mc)
        mc.reset_step()
        probs = router(h, admissible=layer.admissible).detach()
        ref_mc = VirtualMemoryCounter(budget)
        expect = _expected_choice(probs, layer, PredictiveMemoryGate(ref_mc), ref_mc)
        out, chosen = router.route(layer, h, gate, mc, B, S)
        assert chosen in feasible_ok, (seed, chosen)
        assert chosen == expect
        assert mc.peak_bytes <= budget
        assert probs[2] == 0 and probs[4] == 0      # inadmissible experts get zero mass
        seen.add(chosen)
    assert len(seen) >= 2, f"choice never varied: {seen}"


def _cfg(budget):
    cfg = ExperimentConfig()
    cfg.model.num_experts = K
    cfg.memory.budget_bytes = budget
    return cfg


def _vision_step(method, budget):
    # Imported here so that the router tests above run even where torchvision (a
    # module-level import of core_train_loop) is missing; this test then FAILS, visibly.
    from gatedmoe.core_train_loop import train_one_epoch, evaluate
    model = _model(method)
    mc = VirtualMemoryCounter(budget)
    gate = PredictiveMemoryGate(mc)
    debt = [DebtCounter(K) for _ in range(2)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(5)
    loader = [(torch.randn(B, 3, 32, 32), torch.randint(0, 10, (B,)))]
    cfg = _cfg(budget)
    tm, step = train_one_epoch(model, loader, opt, gate, debt, mc, torch.device("cpu"), cfg,
                               variant=method, route_trace=[])
    em = evaluate(model, loader, gate, debt, mc, torch.device("cpu"), cfg)
    return tm, step, em, mc


def test_vision_smoke_gated_and_dense():
    gate0 = PredictiveMemoryGate(VirtualMemoryCounter(10 ** 12))
    m = _model("round_robin")
    shared = gate0.estimate_shared_cost(m.layers[0].shared, B, S)
    costs = [gate0.estimate_expert_cost(e, B, S) for e in m.layers[0].experts]
    budget = shared + costs[3] + 1                 # |F| = 4
    for method in ("gated_top1", "dense"):
        tm, step, em, mc = _vision_step(method, budget)
        assert step == 1
        assert math.isfinite(tm["train_loss"]) and math.isfinite(em["test_loss"])
        if method == "gated_top1":
            assert tm["peak_sram_bytes"] <= budget
        else:
            assert tm["peak_sram_bytes"] > budget  # dense ignores the gate by design
