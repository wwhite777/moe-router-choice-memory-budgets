"""Baseline routing strategies for comparison.

Each baseline wraps a GatedMoEModel and overrides the routing logic.
All use the same VirtualMemoryCounter for fair SRAM accounting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter
from gatedmoe.core_model_define import GatedMoEModel, GatedMoELayer


# ─── Dense baseline ──────────────────────────────────────────────────────────

class DenseRouter:
    """Activates ALL experts every step. Will exceed memory budget."""

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, list]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        expert_outs = []
        activated = []
        for i, expert in enumerate(layer.experts):
            if not adm[i]:
                continue
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)
            mc.load_from_flash(cost)
            expert_outs.append(expert(h))
            activated.append(i)
        if not expert_outs:
            return shared_out, activated
        # Average all expert outputs
        expert_avg = torch.stack(expert_outs).mean(dim=0)
        return shared_out + expert_avg, activated


# ─── Learned Top-1 Router ────────────────────────────────────────────────────

class LearnedTop1Router(nn.Module):
    """Balance-trained top-1 router (NOT task-trained).

    Input: the batch- and token-mean hidden state h.mean(dim=(0, 1)); one routing
    decision per batch and layer. A softmax over the admissible experts ranks them;
    the highest-probability admissible expert that fits the memory gate is chosen.
    The chosen expert's output enters UNWEIGHTED (shared_out + expert_out), so the
    router receives no task-loss gradient: gate_proj is trained only by its
    balancing penalty aux_loss = sum(p^2) (added as 0.01 * aux_loss by the
    training loops). For a task-trained variant see GatedTop1Router.
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.aux_loss = torch.tensor(0.0)

    def forward(self, h: torch.Tensor, admissible=None) -> torch.Tensor:
        # h: (B, S, d) → pool → route
        pooled = h.mean(dim=(0, 1))  # (d,)
        logits = self.gate_proj(pooled)  # (num_experts,)
        if admissible is not None and not all(admissible):
            # Same admissible mask as the gate: softmax over admissible only.
            mask = torch.tensor(admissible, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)

        # Load-balancing: minimize concentration (encourages uniform)
        self.aux_loss = (probs * probs).sum()

        return probs

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, Optional[int]]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        probs = self.forward(h, admissible=adm)
        for chosen in torch.argsort(probs, descending=True).tolist():
            if not adm[chosen]:
                continue
            expert = layer.experts[chosen]
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)

            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                expert_out = expert(h)
                return shared_out + expert_out, chosen
        return shared_out, None


class GatedTop1Router(LearnedTop1Router):
    """Task-trained top-1 router (Amendment 5; Switch-style gate weighting).

    Identical to LearnedTop1Router (same pooled input, same softmax over the
    admissible experts, same choice = highest-probability admissible expert that
    fits the gate, same balancing penalty aux_loss = sum(p^2)), except that the
    chosen expert's output is weighted by its router probability:
    shared_out + probs[chosen] * expert_out. The task loss therefore reaches
    gate_proj. The weighting also scales the expert output by p (< 1 whenever
    more than one expert is admissible).
    """

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, Optional[int]]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        probs = self.forward(h, admissible=adm)
        for chosen in torch.argsort(probs, descending=True).tolist():
            if not adm[chosen]:
                continue
            expert = layer.experts[chosen]
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)

            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                expert_out = expert(h)
                return shared_out + probs[chosen] * expert_out, chosen
        return shared_out, None


# ─── Random Expert ────────────────────────────────────────────────────────────

class RandomRouter:
    """Randomly pick one feasible expert under the current memory budget.

    Uses an ISOLATED RNG stream (own torch.Generator) so its draws never
    perturb model init, data order, or any other method's trajectory.
    """

    def __init__(self, num_experts: int, seed: Optional[int] = None):
        self.num_experts = num_experts
        self.gen = torch.Generator()
        self.gen.manual_seed(seed if seed is not None else 0)

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, Optional[int]]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        feasible = []
        for chosen, expert in enumerate(layer.experts):
            if not adm[chosen]:
                continue
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)
            if mc.can_fit(cost):
                feasible.append((chosen, cost))

        if not feasible:
            return shared_out, None

        idx = torch.randint(0, len(feasible), (1,), generator=self.gen).item()
        chosen, cost = feasible[idx]
        mc.load_from_flash(cost)
        return shared_out + layer.experts[chosen](h), chosen


# ─── Round Robin ──────────────────────────────────────────────────────────────

class RoundRobinRouter:
    """Cycle through experts deterministically."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.current = 0

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, Optional[int]]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        for offset in range(self.num_experts):
            chosen = (self.current + offset) % self.num_experts
            if not adm[chosen]:
                continue
            expert = layer.experts[chosen]
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)

            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                out = shared_out + expert(h)
                self.current = (chosen + 1) % self.num_experts
                return out, chosen

        return shared_out, None


class CycleShuffleRouter:
    """A3.2: fresh random permutation each cycle (isolated RNG). Gap CV sits
    between fixed-permutation (0) and i.i.d. random (high) — the intermediate
    point of the irregularity dose-response."""

    def __init__(self, num_experts: int, seed: Optional[int] = None):
        self.num_experts = num_experts
        self.gen = torch.Generator()
        self.gen.manual_seed(seed if seed is not None else 0)
        self.order = torch.randperm(num_experts, generator=self.gen).tolist()
        self.pos = 0

    def route(self, layer, h, gate, mc, batch_size, seq_len):
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        for offset in range(self.num_experts):
            idx = (self.pos + offset) % self.num_experts
            chosen = self.order[idx]
            if not adm[chosen]:
                continue
            cost = gate.estimate_expert_cost(layer.experts[chosen], batch_size, seq_len)
            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                out = shared_out + layer.experts[chosen](h)
                self.pos = idx + 1
                if self.pos >= self.num_experts:
                    self.order = torch.randperm(
                        self.num_experts, generator=self.gen).tolist()
                    self.pos = 0
                return out, chosen
        return shared_out, None


class FixedCadenceRandomRouter:
    """Cadence-isolation control (LM mechanism intervention, gate G-SM1 §5).

    Draws ONE random expert permutation at construction, then cycles it
    deterministically forever. Cadence regularity is identical to round-robin
    (every expert every K steps); only the ORDER is randomized. Comparing this
    to round-robin (regular, canonical order) and to RandomRouter (irregular,
    i.i.d. draws) isolates whether the LM effect is driven by schedule
    REGULARITY (then this converges like RR) or by the specific RR order (then
    this diverges like random).
    """

    def __init__(self, num_experts: int, seed: Optional[int] = None):
        self.num_experts = num_experts
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else 0)
        self.order = torch.randperm(num_experts, generator=g).tolist()
        self.pos = 0

    def route(self, layer, h, gate, mc, batch_size, seq_len):
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        for offset in range(self.num_experts):
            chosen = self.order[(self.pos + offset) % self.num_experts]
            if not adm[chosen]:
                continue
            cost = gate.estimate_expert_cost(layer.experts[chosen], batch_size, seq_len)
            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                out = shared_out + layer.experts[chosen](h)
                self.pos = (self.order.index(chosen) + 1) % self.num_experts
                return out, chosen
        return shared_out, None


# ─── Mixture of Depths (simplified) ──────────────────────────────────────────

class MixtureOfDepthsRouter(nn.Module):
    """Skip the entire expert layer based on a learned capacity predictor.

    Simplified version: a small MLP decides whether to process or skip.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> bool:
        pooled = h.mean(dim=(0, 1))
        score = torch.sigmoid(self.gate_proj(pooled))
        return score.item() > 0.5

    def route(self, layer: GatedMoELayer, h: torch.Tensor,
              gate: PredictiveMemoryGate, mc: VirtualMemoryCounter,
              batch_size: int, seq_len: int) -> tuple[torch.Tensor, Optional[int]]:
        shared_out = layer.shared(h)
        adm = getattr(layer, "admissible", [True] * len(layer.experts))
        if self.forward(h) and adm[0]:
            # Pick expert 0 (MoD doesn't do per-expert routing)
            expert = layer.experts[0]
            cost = gate.estimate_expert_cost(expert, batch_size, seq_len)
            if mc.can_fit(cost):
                mc.load_from_flash(cost)
                return shared_out + expert(h), 0
        return shared_out, None


# ─── Baseline-aware model forward ────────────────────────────────────────────

class BaselineGatedMoEModel(nn.Module):
    """Wraps GatedMoEModel to use baseline routers instead of debt-based routing."""

    def __init__(self, model: GatedMoEModel, routers: list, method: str):
        super().__init__()
        self.model = model
        self.routers = nn.ModuleList(routers) if isinstance(routers[0], nn.Module) else routers
        self.method = method

    def forward(self, x, gate, debt_counters, mc):
        B = x.shape[0]
        seq_len = self.model.num_patches

        h = self.model.embed(x)

        activated_experts = []
        feasible_sizes = []
        for i, layer in enumerate(self.model.layers):
            # Per-layer memory: allocate shared cost, same as GatedMoEModel
            shared_cost = gate.estimate_shared_cost(layer.shared, B, seq_len)
            mc.allocate(shared_cost)

            residual = h
            h_norm = layer.norm1(h)

            # Feasible-set size at the decision point (before any expert loads),
            # computed identically to GatedMoELayer for all methods.
            adm = getattr(layer, "admissible", [True] * len(layer.experts))
            fsize = sum(
                1 for j, e in enumerate(layer.experts)
                if adm[j] and mc.can_fit(gate.estimate_expert_cost(e, B, seq_len))
            )
            feasible_sizes.append(fsize)

            router = self.routers[i] if i < len(self.routers) else self.routers[0]
            out, activated = router.route(layer, h_norm, gate, mc, B, seq_len)

            h = layer.norm2(residual + out)
            activated_experts.append(activated)

            # Deallocate layer state (same model as GatedMoEModel)
            mc.deallocate(shared_cost)
            if activated is not None:
                if isinstance(activated, list):
                    for a in activated:
                        ec = gate.estimate_expert_cost(layer.experts[a], B, seq_len)
                        mc.deallocate(ec)
                else:
                    ec = gate.estimate_expert_cost(layer.experts[activated], B, seq_len)
                    mc.deallocate(ec)

            # Update debt for tracking stats
            if isinstance(activated, list):
                for a in activated:
                    debt_counters[i].step(a)
            else:
                debt_counters[i].step(activated)

        h = h.mean(dim=1)
        logits = self.model.head(h)
        return logits, {"activated_experts": activated_experts, "feasible_sizes": feasible_sizes}

    def parameters(self, recurse=True):
        yield from self.model.parameters(recurse=recurse)
        for r in self.routers:
            if isinstance(r, nn.Module):
                yield from r.parameters(recurse=recurse)

    def train(self, mode=True):
        self.model.train(mode)
        for r in self.routers:
            if isinstance(r, nn.Module):
                r.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def num_patches(self):
        return self.model.num_patches

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed(self):
        return self.model.embed

    @property
    def head(self):
        return self.model.head


def create_baseline_model(model: GatedMoEModel, method: str,
                          seed: Optional[int] = None,
                          capacity_factor: float = 1.25) -> nn.Module:
    """Factory: wrap model with the appropriate baseline router.

    `seed` feeds RandomRouter's isolated per-layer RNG streams only.
    `capacity_factor` applies to token_capacity only (tuned on validation).
    """
    n_experts = model.num_experts
    n_layers = model.num_layers
    d_model = model.head.in_features

    if method == "dense":
        routers = [DenseRouter() for _ in range(n_layers)]
    elif method == "learned_top1":
        routers = [LearnedTop1Router(d_model, n_experts) for _ in range(n_layers)]
    elif method == "gated_top1":
        routers = [GatedTop1Router(d_model, n_experts) for _ in range(n_layers)]
    elif method == "token_knapsack":
        from gatedmoe.sim_token_routers import TokenTop1KnapsackRouter
        routers = [TokenTop1KnapsackRouter(d_model, n_experts) for _ in range(n_layers)]
    elif method == "token_capacity":
        from gatedmoe.sim_token_routers import TokenTop1CapacityRouter
        routers = [TokenTop1CapacityRouter(d_model, n_experts, capacity_factor)
                   for _ in range(n_layers)]
    elif method == "random":
        base = 2000 + (seed if seed is not None else 0)
        routers = [RandomRouter(n_experts, seed=base + 17 * i) for i in range(n_layers)]
    elif method == "round_robin":
        routers = [RoundRobinRouter(n_experts) for _ in range(n_layers)]
    elif method == "mod":
        routers = [MixtureOfDepthsRouter(d_model) for _ in range(n_layers)]
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    return BaselineGatedMoEModel(model, routers, method)
