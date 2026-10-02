"""E3 — framework-level tensor-memory validation (PROTOCOL.md §7).

Three separately-reported concepts, never merged:
  (1) analytical model bytes: Chat (eq. 3) and Chat+ = Chat + Delta_LN (eq. 4);
  (2) live saved-tensor bytes: unique autograd-saved tensors via
      saved_tensors_hooks (+ params P and grads P analytically);
  (3) framework memory: torch.cuda peak allocated AND reserved, baseline
      context subtracted, peak stats reset per phase, repeated to expose
      workspace variance.

This validates the accounting model at the tensor level. It is NOT hardware
validation and must never be described as such.
"""

import torch
import torch.nn as nn


def chat_bytes(b: int, S: int, d: int, f: int, tau: int = 4) -> int:
    """Eq. (3): Chat = 2P + A for a Linear-ReLU-Linear block."""
    P = tau * (2 * d * f + f + d)
    A = tau * b * S * (2 * d + 2 * f) + b * S * f  # +1-byte ReLU mask
    return 2 * P + A


def delta_ln_bytes(b: int, S: int, d: int, tau: int = 4, scratch: int = 0) -> int:
    """Eq. (4): Delta_LN = 2*tau*b*S*d + 4*tau*b*S + delta_scratch."""
    return 2 * tau * b * S * d + 4 * tau * b * S + scratch


class _Block(nn.Module):
    """Linear-ReLU-Linear with selectable in-place ReLU.

    inplace=True models the MCU implementation Chat assumes (activation
    overwritten, 1-byte mask); inplace=False is what the training code runs.
    """

    def __init__(self, d: int, f: int, inplace: bool):
        super().__init__()
        self.fc1 = nn.Linear(d, f)
        self.act = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Linear(f, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def measure_block(b: int, S: int, d: int, f: int, device: str = "cuda",
                  inplace: bool = False, repeats: int = 3) -> dict:
    """Measure one fwd+bwd of the expert block. Returns bytes per concept."""
    dev = torch.device(device)
    alloc_peaks, reserved_peaks, saved_bytes_list = [], [], []

    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        baseline_alloc = torch.cuda.memory_allocated(dev)

        block = _Block(d, f, inplace).to(dev)
        x = torch.randn(b, S, d, device=dev, requires_grad=False)
        torch.cuda.reset_peak_memory_stats(dev)

        saved = {}

        def pack(t):
            if t.device.type == "cuda":
                saved[(t.data_ptr(), t.nbytes)] = t.nbytes
            return t

        def unpack(t):
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = block(x)
            loss = out.pow(2).mean()
            loss.backward()
        torch.cuda.synchronize(dev)

        alloc_peaks.append(torch.cuda.max_memory_allocated(dev) - baseline_alloc)
        reserved_peaks.append(torch.cuda.max_memory_reserved(dev))
        saved_bytes_list.append(sum(saved.values()))

        del block, x, out, loss
    torch.cuda.empty_cache()

    tau = 4
    P = tau * (2 * d * f + f + d)
    return {
        "b": b, "S": S, "d": d, "f": f, "inplace": inplace,
        "chat": chat_bytes(b, S, d, f),
        "chat_plus": chat_bytes(b, S, d, f) + delta_ln_bytes(b, S, d),
        "measured_alloc_peak": max(alloc_peaks),
        "measured_alloc_peak_all": alloc_peaks,
        "measured_reserved_peak": max(reserved_peaks),
        "live_saved_tensor_bytes": max(saved_bytes_list),
        "live_model_bytes": 2 * P + max(saved_bytes_list),  # params+grads+saved
        "input_bytes": tau * b * S * d,
    }


class _Layer(nn.Module):
    """Full GatedMoE layer path: LN -> shared + expert -> residual -> LN."""

    def __init__(self, d: int, f_shared: int, f_expert: int, inplace: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.shared = _Block(d, f_shared, inplace)
        self.expert = _Block(d, f_expert, inplace)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm1(x)
        out = self.shared(h) + self.expert(h)
        return self.norm2(x + out)


def measure_layer(b: int, S: int, d: int, f_shared: int, f_expert: int,
                  device: str = "cuda", inplace: bool = False,
                  repeats: int = 3) -> dict:
    """Measure one fwd+bwd of the full layer vs Chat_shared+Chat_expert+Delta_LN."""
    dev = torch.device(device)
    alloc_peaks, reserved_peaks = [], []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        baseline_alloc = torch.cuda.memory_allocated(dev)
        layer = _Layer(d, f_shared, f_expert, inplace).to(dev)
        x = torch.randn(b, S, d, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
        layer(x).pow(2).mean().backward()
        torch.cuda.synchronize(dev)
        alloc_peaks.append(torch.cuda.max_memory_allocated(dev) - baseline_alloc)
        reserved_peaks.append(torch.cuda.max_memory_reserved(dev))
        del layer, x
    torch.cuda.empty_cache()
    analytic = (chat_bytes(b, S, d, f_shared) + chat_bytes(b, S, d, f_expert)
                + delta_ln_bytes(b, S, d))
    return {
        "b": b, "S": S, "d": d, "f_shared": f_shared, "f_expert": f_expert,
        "inplace": inplace,
        "analytic_layer_bound": analytic,
        "measured_alloc_peak": max(alloc_peaks),
        "measured_alloc_peak_all": alloc_peaks,
        "measured_reserved_peak": max(reserved_peaks),
    }
