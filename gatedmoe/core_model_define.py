"""GatedMoE model: SharedBase + Experts + Memory-Gated Routing.

Architecture: patch embedding → N GatedMoE layers → classification head.
Each layer: always-on shared FFN + at most one routed expert (residual).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from gatedmoe import core_gate_memory as gate_mod
from gatedmoe import core_debt_scheduler as debt_mod


# ─── Building blocks ─────────────────────────────────────────────────────────

class SharedBase(nn.Module):
    """Always-on shared expert. Captures common patterns."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Expert(nn.Module):
    """One routed expert. Learns residual on top of shared base."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ─── GatedMoE Layer ──────────────────────────────────────────────────────────

class GatedMoELayer(nn.Module):
    """One MoE layer: shared base + memory-gated expert promotion."""

    def __init__(
        self,
        d_model: int,
        d_ff_shared: int,
        d_ff_expert,  # int or list[int] for heterogeneous experts
        num_experts: int,
        admissible_mask=None,  # list[int] of admissible expert ids, or None = all
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.shared = SharedBase(d_model, d_ff_shared)
        if isinstance(d_ff_expert, (list, tuple)):
            assert len(d_ff_expert) == num_experts
            self.experts = nn.ModuleList(
                [Expert(d_model, d_ff_expert[i]) for i in range(num_experts)]
            )
        else:
            self.experts = nn.ModuleList(
                [Expert(d_model, d_ff_expert) for _ in range(num_experts)]
            )
        self.norm2 = nn.LayerNorm(d_model)
        self.num_experts = num_experts
        # S1 controlled choice-set study: gate may only admit masked experts.
        if admissible_mask is None:
            self.admissible = [True] * num_experts
        else:
            self.admissible = [i in set(admissible_mask) for i in range(num_experts)]

    def forward(
        self,
        x: torch.Tensor,
        gate: gate_mod.PredictiveMemoryGate,
        debt: debt_mod.DebtCounter,
        mc: gate_mod.VirtualMemoryCounter,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, Optional[int], int]:
        residual = x
        h = self.norm1(x)

        # Shared base always runs
        shared_out = self.shared(h)

        # Memory gate: pick the highest-debt expert among those that fit.
        expert_costs = [
            gate.estimate_expert_cost(expert, batch_size, seq_len)
            for expert in self.experts
        ]
        feasible_mask = torch.tensor(
            [adm and mc.can_fit(cost)
             for adm, cost in zip(self.admissible, expert_costs)],
            dtype=torch.bool,
            device=debt.debt.device,
        )
        feasible_size = int(feasible_mask.sum().item())

        if feasible_mask.any():
            candidate = debt.select_expert(feasible_mask)
            expert = self.experts[candidate]
            expert_cost = expert_costs[candidate]
            mc.load_from_flash(expert_cost)
            expert_out = expert(h)
            out = shared_out + expert_out
            activated = candidate
        else:
            out = shared_out
            activated = None

        out = self.norm2(residual + out)
        return out, activated, feasible_size


# ─── Patch Embedding ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, d_model)
        x = self.proj(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed
        return x


# ─── Full Model ──────────────────────────────────────────────────────────────

class GatedMoEModel(nn.Module):
    """Full GatedMoE classifier: patch embed → N layers → pool → head."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff_shared: int = 256,
        d_ff_expert = 256,  # int or list[int] for heterogeneous
        num_experts: int = 4,
        num_layers: int = 4,
        num_classes: int = 10,
        input_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        admissible_mask=None,
    ):
        super().__init__()
        self.embed = PatchEmbedding(input_channels, d_model, patch_size, image_size)
        self.layers = nn.ModuleList([
            GatedMoELayer(d_model, d_ff_shared, d_ff_expert, num_experts,
                          admissible_mask=admissible_mask)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)
        self.num_patches = self.embed.num_patches
        self.num_experts = num_experts
        self.num_layers = num_layers

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        gate: gate_mod.PredictiveMemoryGate,
        debt_counters: List[debt_mod.DebtCounter],
        mc: gate_mod.VirtualMemoryCounter,
    ) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        seq_len = self.num_patches

        # Patch embedding
        h = self.embed(x)

        # Pass through layers (per-layer memory allocation/deallocation)
        activated_experts = []
        feasible_sizes = []
        for i, layer in enumerate(self.layers):
            # Allocate shared base cost for this layer
            shared_cost = gate.estimate_shared_cost(layer.shared, B, seq_len)
            mc.allocate(shared_cost)
            h, activated, fsize = layer(h, gate, debt_counters[i], mc, B, seq_len)
            activated_experts.append(activated)
            feasible_sizes.append(fsize)
            # Deallocate: previous layer's state freed after next layer reads it
            mc.deallocate(shared_cost)
            if activated is not None:
                expert_cost = gate.estimate_expert_cost(layer.experts[activated], B, seq_len)
                mc.deallocate(expert_cost)

        # Global average pool → classify
        h = h.mean(dim=1)
        logits = self.head(h)

        info = {"activated_experts": activated_experts, "feasible_sizes": feasible_sizes}
        return logits, info

    def get_expert_params_flat(self, layer_idx: int) -> List[torch.Tensor]:
        """Get flattened parameter vectors for each expert in a layer."""
        layer = self.layers[layer_idx]
        result = []
        for expert in layer.experts:
            flat = torch.cat([p.flatten() for p in expert.parameters()])
            result.append(flat)
        return result

    def get_shared_grad_flat(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get flattened gradient of the shared base in a layer."""
        layer = self.layers[layer_idx]
        grads = []
        for p in layer.shared.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        if not grads:
            return None
        return torch.cat(grads)


DEFAULT_HETERO_EXPERTS = [128, 256, 512, 1024, 128, 256, 512, 1024]  # cycles for K=8

def build_model_from_config(cfg) -> GatedMoEModel:
    """Construct model from an ExperimentConfig."""
    m = cfg.model
    d_ff_expert = m.d_ff_expert
    if d_ff_expert is None:
        # Build a list of length num_experts by cycling through base sizes
        base = [128, 256, 512, 1024]
        d_ff_expert = [base[i % len(base)] for i in range(m.num_experts)]
    return GatedMoEModel(
        d_model=m.d_model,
        d_ff_shared=m.d_ff_shared,
        d_ff_expert=d_ff_expert,
        num_experts=m.num_experts,
        num_layers=m.num_layers,
        num_classes=m.num_classes,
        input_channels=m.input_channels,
        image_size=m.image_size,
        patch_size=m.patch_size,
        admissible_mask=getattr(m, "admissible_mask", None),
    )
