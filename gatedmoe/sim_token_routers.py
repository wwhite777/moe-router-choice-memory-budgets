"""E5 — token-level routing arms (PROTOCOL.md).

TokenTop1KnapsackRouter: per-token top-1 routing where the memory gate becomes
a greedy knapsack over co-activated experts. An admitted expert's cost is
computed for the ACTUAL number of tokens it processes via the same analytical
gate (estimate_expert_cost with batch=1, seq_len=T_e); experts are admitted in
decreasing routed-score-mass order while the budget allows; tokens of
non-admitted experts fall back to the shared path.

TokenTop1CapacityRouter: the standard MoE control — top-1 with a capacity
factor, NO memory gate. Its memory use is recorded by the same virtual counter
so budget violations are measured and reported (reference line, like Dense).

Flash-traffic convention: like the batch-level arms, the FULL admitted cost is
booked via load_from_flash for cross-arm consistency; flash bytes therefore
overcount pure parameter traffic identically in every arm (analysis note).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_model_define import GatedMoELayer


class _TokenRouterBase(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.aux_loss = torch.tensor(0.0)

    def _token_probs(self, h: torch.Tensor, admissible) -> torch.Tensor:
        logits = self.gate_proj(h)  # (B, S, K)
        if admissible is not None and not all(admissible):
            mask = torch.tensor(admissible, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, float("-inf"))
        return F.softmax(logits, dim=-1)

    def _load_balance_aux(self, probs: torch.Tensor, top1: torch.Tensor) -> torch.Tensor:
        # Switch-style: K * sum_e frac_tokens_e * mean_prob_e
        K = self.num_experts
        frac = torch.zeros(K, device=probs.device)
        for e in range(K):
            frac[e] = (top1 == e).float().mean()
        mean_prob = probs.mean(dim=(0, 1))
        return K * (frac * mean_prob).sum()

    @staticmethod
    def _scatter_expert_outputs(layer, h, shared_out, top1, experts_to_run):
        """out = shared + expert_e(h) on each token routed to an admitted e."""
        out = shared_out.clone()
        for e, token_mask in experts_to_run:
            idx = token_mask.nonzero(as_tuple=True)
            out[idx] = out[idx] + layer.experts[e](h[idx])
        return out


class TokenTop1KnapsackRouter(_TokenRouterBase):
    """Greedy score-mass knapsack under the SRAM budget."""

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int):
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        probs = self._token_probs(h, adm)  # (B, S, K)
        top1 = probs.argmax(dim=-1)  # (B, S)
        self.aux_loss = self._load_balance_aux(probs, top1)

        # Score mass and token count per expert
        order = []
        for e in range(self.num_experts):
            if not adm[e]:
                continue
            token_mask = top1 == e
            T_e = int(token_mask.sum().item())
            if T_e == 0:
                continue
            mass = float(probs[..., e][token_mask].sum().item())
            order.append((mass, e, token_mask, T_e))
        order.sort(key=lambda x: -x[0])

        admitted = []
        experts_to_run = []
        for mass, e, token_mask, T_e in order:
            cost = gate.estimate_expert_cost(layer.experts[e], 1, T_e)
            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                admitted.append(e)
                experts_to_run.append((e, token_mask))

        if not experts_to_run:
            return shared_out, admitted
        out = self._scatter_expert_outputs(layer, h, shared_out, top1, experts_to_run)
        return out, admitted


class TokenTop1CapacityRouter(_TokenRouterBase):
    """Standard top-1 + capacity factor; no memory gate (reference line)."""

    def __init__(self, d_model: int, num_experts: int, capacity_factor: float = 1.25):
        super().__init__(d_model, num_experts)
        self.capacity_factor = capacity_factor

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int):
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        probs = self._token_probs(h, adm)
        top1 = probs.argmax(dim=-1)
        self.aux_loss = self._load_balance_aux(probs, top1)

        n_tokens = top1.numel()
        cap = max(1, math.ceil(self.capacity_factor * n_tokens / self.num_experts))

        admitted = []
        experts_to_run = []
        for e in range(self.num_experts):
            if not adm[e]:
                continue
            token_mask = top1 == e
            T_e = int(token_mask.sum().item())
            if T_e == 0:
                continue
            if T_e > cap:
                # Keep the cap highest-probability tokens; rest overflow to shared.
                scores = probs[..., e].masked_fill(~token_mask, -1.0)
                flat = scores.flatten()
                keep = torch.topk(flat, cap).indices
                new_mask = torch.zeros_like(flat, dtype=torch.bool)
                new_mask[keep] = True
                token_mask = new_mask.view_as(token_mask)
                T_e = cap
            cost = gate.estimate_expert_cost(layer.experts[e], 1, T_e)
            mc.load_from_flash(cost)  # recorded even when it violates the budget
            admitted.append(e)
            experts_to_run.append((e, token_mask))

        if not experts_to_run:
            return shared_out, admitted
        out = self._scatter_expert_outputs(layer, h, shared_out, top1, experts_to_run)
        return out, admitted
