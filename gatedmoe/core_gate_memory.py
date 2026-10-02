"""Predictive memory gate and virtual SRAM counter.

The intellectual core of GatedMoE: decides whether to promote an expert
based on estimated SRAM cost, without running the forward pass first.
"""

import torch
import torch.nn as nn


class VirtualMemoryCounter:
    """Simulates SRAM usage on an MCU. Tracks live bytes, peak, and Flash traffic."""

    def __init__(self, budget_bytes: int, dtype_bytes: int = 4):
        self.budget = budget_bytes
        self.dtype_bytes = dtype_bytes
        self.live_bytes = 0
        self.peak_bytes = 0
        self.flash_to_sram_bytes = 0
        self._step_peak = 0

    def allocate(self, nbytes: int) -> None:
        self.live_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self._step_peak = max(self._step_peak, self.live_bytes)

    def deallocate(self, nbytes: int) -> None:
        self.live_bytes = max(0, self.live_bytes - nbytes)

    def load_from_flash(self, nbytes: int) -> None:
        self.flash_to_sram_bytes += nbytes
        self.allocate(nbytes)

    def can_fit(self, nbytes: int) -> bool:
        return (self.live_bytes + nbytes) <= self.budget

    def reset_step(self) -> int:
        """Reset per-step state. Returns step peak before reset."""
        peak = self._step_peak
        self._step_peak = 0
        self.live_bytes = 0
        return peak

    def get_stats(self) -> dict:
        return {
            "peak_sram_bytes": self.peak_bytes,
            "flash_to_sram_bytes": self.flash_to_sram_bytes,
            "current_live_bytes": self.live_bytes,
            "budget_bytes": self.budget,
        }


class PredictiveMemoryGate:
    """Decides expert promotion based on estimated SRAM cost.

    For standard FFN experts (Linear-ReLU-Linear), the cost is deterministic
    from (architecture, batch_size, seq_len, dtype). No forward pass needed.
    """

    def __init__(self, memory_counter: VirtualMemoryCounter, dtype_bytes: int = 4):
        self.mc = memory_counter
        self.dtype_bytes = dtype_bytes
        self._cost_cache: dict[tuple, int] = {}

    def estimate_expert_cost(self, expert: nn.Module, batch_size: int, seq_len: int) -> int:
        """Analytically compute bytes needed to activate an expert.

        For Linear(in, out) in forward+backward:
          - params: in*out + out (weights + bias)
          - saved activation (for grad_weight): B*S*in
          - output activation: B*S*out
          - ReLU mask: B*S*out (stored as uint8 = 1 byte each)

        Returns total bytes.
        """
        # Content-based key: cost depends only on layer shapes + (b, S), never
        # on object identity (id() can be reused after garbage collection).
        sig = []
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                sig.append(("lin", module.in_features, module.out_features))
            elif isinstance(module, (nn.ReLU, nn.GELU)):
                sig.append(("act",))
        cache_key = (tuple(sig), batch_size, seq_len)
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]

        param_bytes = 0
        act_bytes = 0

        prev_out = None
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features
                # Parameter memory
                param_bytes += (in_f * out_f + out_f) * self.dtype_bytes
                # Saved input activation for backward
                act_bytes += batch_size * seq_len * in_f * self.dtype_bytes
                # Output activation
                act_bytes += batch_size * seq_len * out_f * self.dtype_bytes
                prev_out = out_f
            elif isinstance(module, (nn.ReLU, nn.GELU)):
                if prev_out is not None:
                    # ReLU mask stored as bytes
                    act_bytes += batch_size * seq_len * prev_out

        # Gradient buffers (same size as params)
        grad_bytes = param_bytes

        total = param_bytes + act_bytes + grad_bytes
        self._cost_cache[cache_key] = total
        return total

    def estimate_shared_cost(self, shared: nn.Module, batch_size: int, seq_len: int) -> int:
        """Same estimation for the shared base path."""
        return self.estimate_expert_cost(shared, batch_size, seq_len)

    def should_promote(self, expert: nn.Module, batch_size: int, seq_len: int) -> bool:
        """The core gate: can we afford this expert within SRAM budget?"""
        cost = self.estimate_expert_cost(expert, batch_size, seq_len)
        return self.mc.can_fit(cost)

    def minimum_useful_budget(self, shared: nn.Module, experts: nn.ModuleList,
                               batch_size: int, seq_len: int) -> int:
        """Minimum budget where at least one expert can be promoted."""
        shared_cost = self.estimate_shared_cost(shared, batch_size, seq_len)
        min_expert_cost = min(
            self.estimate_expert_cost(e, batch_size, seq_len) for e in experts
        )
        return shared_cost + min_expert_cost
