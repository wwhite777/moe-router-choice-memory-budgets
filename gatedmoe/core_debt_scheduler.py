"""Expert debt counters for GatedMoE routing.

DebtCounter: uniform +1 increment (GatedMoE-Base).
GradientDebtCounter: gradient-informed increment (GatedMoE-G).
"""

import torch
import torch.nn as nn
from typing import Optional, List


class DebtCounter:
    """GatedMoE-Base: each sleeping expert accumulates +1 debt per step."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.debt = torch.zeros(num_experts)
        self.total_activations = torch.zeros(num_experts, dtype=torch.long)
        self.steps_since_active = torch.zeros(num_experts, dtype=torch.long)
        self.max_sleep_length = torch.zeros(num_experts, dtype=torch.long)

    def select_expert(self, feasible_mask: Optional[torch.Tensor] = None) -> int:
        """Pick the expert with highest debt among feasible ones."""
        if feasible_mask is not None:
            masked = self.debt.clone()
            masked[~feasible_mask] = -float("inf")
            return masked.argmax().item()
        return self.debt.argmax().item()

    def step(self, activated_idx: Optional[int]) -> None:
        """Update debt after a training step."""
        self.debt += 1.0
        self.steps_since_active += 1

        if activated_idx is not None:
            self.debt[activated_idx] = 0.0
            self.max_sleep_length[activated_idx] = max(
                self.max_sleep_length[activated_idx].item(),
                self.steps_since_active[activated_idx].item(),
            )
            self.steps_since_active[activated_idx] = 0
            self.total_activations[activated_idx] += 1

    def get_stats(self) -> dict:
        return {
            "debt": self.debt.tolist(),
            "total_activations": self.total_activations.tolist(),
            "steps_since_active": self.steps_since_active.tolist(),
            "max_sleep_length": self.max_sleep_length.tolist(),
            "utilization_entropy": self._utilization_entropy(),
        }

    def _utilization_entropy(self) -> float:
        counts = self.total_activations.float()
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    def reset(self) -> None:
        self.debt.zero_()
        self.total_activations.zero_()
        self.steps_since_active.zero_()
        self.max_sleep_length.zero_()


class GradientDebtCounter(DebtCounter):
    """GatedMoE-G: debt increment weighted by historical loss reduction.

    When an expert is active, record its loss reduction. Sleeping experts
    accumulate debt proportional to their EMA of past loss reductions —
    experts that helped more get promoted sooner.
    """

    def __init__(self, num_experts: int, ema_alpha: float = 0.3):
        super().__init__(num_experts)
        self.loss_ema = torch.ones(num_experts)  # init to 1 (equal priority)
        self.ema_alpha = ema_alpha
        self._prev_loss = None

    def step(
        self,
        activated_idx: Optional[int],
        current_loss: Optional[float] = None,
        **kwargs,  # accept and ignore old args for compatibility
    ) -> None:
        # Update loss EMA for the active expert
        if activated_idx is not None and current_loss is not None and self._prev_loss is not None:
            delta = max(self._prev_loss - current_loss, 0.0)  # loss reduction (positive = good)
            self.loss_ema[activated_idx] = (
                (1 - self.ema_alpha) * self.loss_ema[activated_idx] + self.ema_alpha * delta
            )
        if current_loss is not None:
            self._prev_loss = current_loss

        # Sleeping experts accumulate debt weighted by their historical utility
        for i in range(self.num_experts):
            if i == activated_idx:
                continue
            self.debt[i] += self.loss_ema[i].item()

        self.steps_since_active += 1
        if activated_idx is not None:
            self.debt[activated_idx] = 0.0
            self.max_sleep_length[activated_idx] = max(
                self.max_sleep_length[activated_idx].item(),
                self.steps_since_active[activated_idx].item(),
            )
            self.steps_since_active[activated_idx] = 0
            self.total_activations[activated_idx] += 1
