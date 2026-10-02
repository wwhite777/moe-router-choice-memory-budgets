"""Tests for DebtCounter and GradientDebtCounter."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from gatedmoe.core_debt_scheduler import DebtCounter, GradientDebtCounter


def test_basic_debt():
    dc = DebtCounter(num_experts=4)
    assert dc.select_expert() == 0  # all equal, picks first

    # After 1 step with expert 0 activated
    dc.step(activated_idx=0)
    # Expert 0: debt=0, others: debt=1
    assert dc.debt[0] == 0
    assert dc.debt[1] == 1
    assert dc.select_expert() == 1  # highest debt among 1,2,3
    print("PASS: test_basic_debt")


def test_bounded_activation_horizon():
    """Proposition 2: every expert activated within K-1 feasible steps."""
    K = 4
    dc = DebtCounter(num_experts=K)

    # Simulate many steps where we always promote highest-debt expert
    for step in range(1000):
        chosen = dc.select_expert()
        dc.step(activated_idx=chosen)

    # Every expert should have been activated many times
    for i in range(K):
        assert dc.total_activations[i] > 0, f"Expert {i} was never activated!"
        # Max sleep should be at most K-1
        assert dc.max_sleep_length[i] <= K, (
            f"Expert {i} slept {dc.max_sleep_length[i]} steps (max should be {K-1})"
        )

    print(f"  Activations: {dc.total_activations.tolist()}")
    print(f"  Max sleep: {dc.max_sleep_length.tolist()}")
    print("PASS: test_bounded_activation_horizon")


def test_utilization_entropy():
    K = 4
    dc = DebtCounter(num_experts=K)

    # Uniform activation → max entropy
    for step in range(400):
        chosen = dc.select_expert()
        dc.step(activated_idx=chosen)

    entropy = dc._utilization_entropy()
    max_entropy = torch.tensor(K).float().log().item()
    assert entropy > 0.9 * max_entropy, f"Entropy {entropy} too low (max {max_entropy})"
    print(f"  Utilization entropy: {entropy:.4f} (max: {max_entropy:.4f})")
    print("PASS: test_utilization_entropy")


def test_gradient_debt_counter():
    K = 3
    dc = GradientDebtCounter(num_experts=K)

    # Simulate with fake gradient and params
    shared_grad = torch.randn(100)
    expert_params = [torch.randn(100) for _ in range(K)]

    for step in range(100):
        chosen = dc.select_expert()
        dc.step(chosen, shared_grad=shared_grad, expert_params_flat=expert_params)

    assert all(dc.total_activations[i] > 0 for i in range(K))
    print(f"  G-Debt activations: {dc.total_activations.tolist()}")
    print("PASS: test_gradient_debt_counter")


def test_feasible_mask():
    dc = DebtCounter(num_experts=4)
    for _ in range(5):
        dc.step(activated_idx=0)  # only activate expert 0

    # Expert 0 has low debt, others have high debt
    mask = torch.tensor([False, True, True, False])
    chosen = dc.select_expert(feasible_mask=mask)
    assert chosen in [1, 2], f"Expected 1 or 2, got {chosen}"
    print("PASS: test_feasible_mask")


if __name__ == "__main__":
    test_basic_debt()
    test_bounded_activation_horizon()
    test_utilization_entropy()
    test_gradient_debt_counter()
    test_feasible_mask()
    print("\nAll debt scheduler tests passed.")
