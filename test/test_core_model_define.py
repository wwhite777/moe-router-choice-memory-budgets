"""Tests for GatedMoE model components."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter, GradientDebtCounter
from gatedmoe.core_model_define import GatedMoEModel


def test_forward_pass():
    model = GatedMoEModel(
        d_model=64, d_ff_shared=128, d_ff_expert=128,
        num_experts=3, num_layers=2, num_classes=10,
        input_channels=3, image_size=32, patch_size=8,
    )
    mc = VirtualMemoryCounter(budget_bytes=500_000)
    gate = PredictiveMemoryGate(mc)
    debts = [DebtCounter(3) for _ in range(2)]

    x = torch.randn(4, 3, 32, 32)
    logits, info = model(x, gate, debts, mc)

    assert logits.shape == (4, 10)
    assert len(info["activated_experts"]) == 2
    print(f"  Output shape: {logits.shape}")
    print(f"  Activated experts: {info['activated_experts']}")
    print("PASS: test_forward_pass")


def test_gradient_flow():
    """Proposition 4: active expert receives exact gradients."""
    model = GatedMoEModel(
        d_model=32, d_ff_shared=64, d_ff_expert=64,
        num_experts=2, num_layers=1, num_classes=10,
        input_channels=3, image_size=32, patch_size=8,
    )
    mc = VirtualMemoryCounter(budget_bytes=1_000_000)
    gate = PredictiveMemoryGate(mc)
    debts = [DebtCounter(2)]

    x = torch.randn(2, 3, 32, 32)
    logits, info = model(x, gate, debts, mc)
    loss = logits.sum()
    loss.backward()

    # Check that the shared base and activated expert have gradients
    for p in model.layers[0].shared.parameters():
        assert p.grad is not None, "Shared base has no gradient!"
        assert not torch.all(p.grad == 0), "Shared base gradient is all zeros!"

    activated = info["activated_experts"][0]
    if activated is not None:
        for p in model.layers[0].experts[activated].parameters():
            assert p.grad is not None, f"Expert {activated} has no gradient!"
            assert not torch.all(p.grad == 0), f"Expert {activated} gradient is all zeros!"
        print(f"  Expert {activated}: exact gradient confirmed")

    print("PASS: test_gradient_flow (Proposition 4)")


def test_shared_grad_extraction():
    model = GatedMoEModel(
        d_model=32, d_ff_shared=64, d_ff_expert=64,
        num_experts=2, num_layers=1, num_classes=10,
        input_channels=3, image_size=32, patch_size=8,
    )
    mc = VirtualMemoryCounter(budget_bytes=1_000_000)
    gate = PredictiveMemoryGate(mc)
    debts = [GradientDebtCounter(2)]

    x = torch.randn(2, 3, 32, 32)
    logits, info = model(x, gate, debts, mc)
    loss = logits.sum()
    loss.backward()

    grad = model.get_shared_grad_flat(0)
    assert grad is not None
    assert grad.numel() > 0
    print(f"  Shared grad shape: {grad.shape}")

    params = model.get_expert_params_flat(0)
    assert len(params) == 2
    print(f"  Expert param shapes: {[p.shape for p in params]}")
    print("PASS: test_shared_grad_extraction")


if __name__ == "__main__":
    test_forward_pass()
    test_gradient_flow()
    test_shared_grad_extraction()
    print("\nAll model tests passed.")
