"""SRAM / Flash / MACs profiler for simulated MCU evaluation.

Reports hardware-agnostic metrics that don't require a physical board.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    peak_sram_bytes: int = 0
    flash_to_sram_bytes: int = 0
    macs: int = 0
    active_expert_ids: list = field(default_factory=list)
    budget_utilization: float = 0.0


def count_macs_linear(in_features: int, out_features: int, batch_size: int, seq_len: int) -> int:
    """MACs for a Linear layer: B * S * in * out."""
    return batch_size * seq_len * in_features * out_features


def count_macs_module(module: nn.Module, batch_size: int, seq_len: int) -> int:
    """Count total MACs for a module (only Linear layers contribute significantly)."""
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += count_macs_linear(m.in_features, m.out_features, batch_size, seq_len)
    return total


def profile_model_step(
    model,
    activated_experts: list,
    batch_size: int,
    mc,
) -> StepMetrics:
    """Compute all metrics for one training step."""
    seq_len = model.num_patches

    # MACs: shared base always runs, plus activated experts
    macs = 0
    for layer in model.layers:
        macs += count_macs_module(layer.shared, batch_size, seq_len)
        macs += count_macs_module(layer.norm1, batch_size, seq_len)
        macs += count_macs_module(layer.norm2, batch_size, seq_len)

    for i, expert_idx in enumerate(activated_experts):
        if expert_idx is None:
            continue
        # Dense baseline returns list of all activated experts
        indices = expert_idx if isinstance(expert_idx, list) else [expert_idx]
        for idx in indices:
            expert = model.layers[i].experts[idx]
            macs += count_macs_module(expert, batch_size, seq_len)

    # Embedding (Conv2d as Linear equivalent: in_ch*P*P → d_model per patch)
    proj = model.embed.proj
    embed_macs = batch_size * model.num_patches * (proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1]) * proj.out_channels
    macs += embed_macs
    # Classification head
    macs += count_macs_module(model.head, batch_size, 1)

    stats = mc.get_stats()

    return StepMetrics(
        peak_sram_bytes=stats["peak_sram_bytes"],
        flash_to_sram_bytes=stats["flash_to_sram_bytes"],
        macs=macs,
        active_expert_ids=activated_experts,
        budget_utilization=stats["peak_sram_bytes"] / max(stats["budget_bytes"], 1),
    )


def validate_memory_estimator(expert: nn.Module, batch_size: int, seq_len: int,
                                gate, dtype_bytes: int = 4) -> dict:
    """Hook-based validation: compare analytical estimate vs actual tensor sizes.

    Used only in testing, not during training.
    """
    estimated = gate.estimate_expert_cost(expert, batch_size, seq_len)

    # Measure actual sizes via hooks
    actual_bytes = 0
    handles = []

    def hook_fn(module, input, output):
        nonlocal actual_bytes
        if isinstance(input, tuple):
            for t in input:
                if isinstance(t, torch.Tensor):
                    actual_bytes += t.nelement() * t.element_size()
        if isinstance(output, torch.Tensor):
            actual_bytes += output.nelement() * output.element_size()

    for m in expert.modules():
        if isinstance(m, (nn.Linear, nn.ReLU, nn.GELU)):
            handles.append(m.register_forward_hook(hook_fn))

    dummy = torch.randn(batch_size, seq_len, expert.fc1.in_features)
    with torch.no_grad():
        expert(dummy)

    for h in handles:
        h.remove()

    # Add param + grad bytes for fair comparison
    param_bytes = sum(p.nelement() * p.element_size() for p in expert.parameters()) * 2

    return {
        "estimated_bytes": estimated,
        "measured_activation_bytes": actual_bytes,
        "param_grad_bytes": param_bytes,
        "total_measured": actual_bytes + param_bytes,
    }
