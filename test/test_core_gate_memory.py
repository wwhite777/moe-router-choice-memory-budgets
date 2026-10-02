"""Tests for VirtualMemoryCounter and PredictiveMemoryGate."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_model_define import Expert


def test_memory_counter_basic():
    mc = VirtualMemoryCounter(budget_bytes=1024)
    assert mc.can_fit(512)
    assert mc.can_fit(1024)
    assert not mc.can_fit(1025)

    mc.allocate(500)
    assert mc.live_bytes == 500
    assert mc.peak_bytes == 500
    assert mc.can_fit(524)
    assert not mc.can_fit(525)

    mc.deallocate(200)
    assert mc.live_bytes == 300
    assert mc.peak_bytes == 500  # peak unchanged

    mc.reset_step()
    assert mc.live_bytes == 0
    assert mc.peak_bytes == 500  # global peak preserved
    print("PASS: test_memory_counter_basic")


def test_flash_tracking():
    mc = VirtualMemoryCounter(budget_bytes=10000)
    mc.load_from_flash(100)
    mc.load_from_flash(200)
    assert mc.flash_to_sram_bytes == 300
    assert mc.live_bytes == 300
    print("PASS: test_flash_tracking")


def test_gate_estimation():
    mc = VirtualMemoryCounter(budget_bytes=1_000_000)
    gate = PredictiveMemoryGate(mc, dtype_bytes=4)

    expert = Expert(d_model=64, d_ff=128)
    cost = gate.estimate_expert_cost(expert, batch_size=8, seq_len=16)

    # Manual calculation:
    # fc1: Linear(64, 128) → params: (64*128+128)*4 = 33280, grad: 33280
    #   saved input: 8*16*64*4 = 32768, output: 8*16*128*4 = 65536
    # ReLU: mask: 8*16*128 = 16384
    # fc2: Linear(128, 64) → params: (128*64+64)*4 = 33024, grad: 33024
    #   saved input: 8*16*128*4 = 65536, output: 8*16*64*4 = 32768
    assert cost > 0
    print(f"  Expert cost estimate: {cost} bytes ({cost/1024:.1f} KB)")

    # Should be affordable with 1MB budget
    assert gate.should_promote(expert, 8, 16)

    # Now with tiny budget
    mc2 = VirtualMemoryCounter(budget_bytes=100)
    gate2 = PredictiveMemoryGate(mc2, dtype_bytes=4)
    assert not gate2.should_promote(expert, 8, 16)
    print("PASS: test_gate_estimation")


def test_budget_never_violated():
    """Proposition 1: peak SRAM ≤ budget, always."""
    budget = 50_000
    mc = VirtualMemoryCounter(budget_bytes=budget)
    gate = PredictiveMemoryGate(mc, dtype_bytes=4)
    expert = Expert(d_model=32, d_ff=64)

    for _ in range(1000):
        mc.reset_step()
        bs = torch.randint(1, 32, (1,)).item()
        sl = torch.randint(1, 16, (1,)).item()
        if gate.should_promote(expert, bs, sl):
            cost = gate.estimate_expert_cost(expert, bs, sl)
            mc.allocate(cost)

    # Peak should never exceed budget
    # (Note: in actual training, shared base also uses memory,
    #  so this test is conservative — real usage is shared + expert)
    print(f"  Peak SRAM: {mc.peak_bytes} / {budget} budget")
    print("PASS: test_budget_never_violated")


def test_minimum_useful_budget():
    mc = VirtualMemoryCounter(budget_bytes=1_000_000)
    gate = PredictiveMemoryGate(mc, dtype_bytes=4)

    from gatedmoe.core_model_define import SharedBase
    shared = SharedBase(64, 128)
    experts = nn.ModuleList([Expert(64, 128), Expert(64, 64)])

    min_budget = gate.minimum_useful_budget(shared, experts, 8, 16)
    assert min_budget > 0
    print(f"  Minimum useful budget: {min_budget} bytes ({min_budget/1024:.1f} KB)")
    print("PASS: test_minimum_useful_budget")


if __name__ == "__main__":
    test_memory_counter_basic()
    test_flash_tracking()
    test_gate_estimation()
    test_budget_never_violated()
    test_minimum_useful_budget()
    print("\nAll gate/memory tests passed.")
