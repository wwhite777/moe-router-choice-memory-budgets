"""E6-LM — tiny MoE-transformer language model (PROTOCOL.md).

Decoder-only causal LM whose FFN sublayer is the GatedMoE construction
(always-on shared FFN + gate-admitted expert). LMBlock exposes the SAME
attribute surface as GatedMoELayer (.shared, .experts, .admissible, .norm1,
.norm2) so the batch-level routers in sim_baseline_routers reuse unchanged.

Accounting model: per layer the backbone cost (attention + shared FFN,
forward+backward, G=P) is allocated first; experts are gated against the
remaining budget exactly as in the vision models. Attention cost:
  P_attn = tau * 4 * d^2  (+ biases)         A_attn = tau*b*S*(6d) + tau*b*h*S^2
(q,k,v,out projections' saved inputs/outputs collapsed to 6d per token; one
h-head S x S probability tensor saved for backward).
"""

import math
import os
import io
import zipfile
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatedmoe.core_model_define import SharedBase, Expert
from gatedmoe import core_gate_memory as gate_mod
from gatedmoe import core_debt_scheduler as debt_mod

WIKITEXT2_URL = "https://wikitext.smerity.com/wikitext-2-v1.zip"


# ─── Data ────────────────────────────────────────────────────────────────────

def load_wikitext2(data_dir: str = "./data"):
    """Word-level WikiText-2. Returns (train, valid, test) LongTensors + vocab size.

    Downloads the canonical zip (author's mirror) on first use; md5 recorded.
    """
    root = Path(data_dir) / "wikitext-2"
    zpath = Path(data_dir) / "wikitext-2-v1.zip"
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        if not zpath.exists():
            urllib.request.urlretrieve(WIKITEXT2_URL, zpath)
        md5 = hashlib.md5(open(zpath, "rb").read()).hexdigest()
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root.parent)
        (root / "DOWNLOAD_PROVENANCE.txt").write_text(
            f"source={WIKITEXT2_URL}\nmd5={md5}\n")

    def tokens(split):
        p = root / f"wiki.{split}.tokens"
        toks = []
        for line in open(p, encoding="utf-8"):
            toks.extend(line.split() + ["<eos>"])
        return toks

    train_t = tokens("train")
    vocab = {"<unk>": 0}
    for t in train_t:
        if t not in vocab:
            vocab[t] = len(vocab)

    def encode(toks):
        unk = vocab["<unk>"]
        return torch.tensor([vocab.get(t, unk) for t in toks], dtype=torch.long)

    return (encode(train_t), encode(tokens("valid")), encode(tokens("test")),
            len(vocab))


def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Classic contiguous batching: (batch, n_steps). Order is deterministic,
    so batch order is matched across methods by construction."""
    n = data.numel() // batch_size
    return data[: n * batch_size].view(batch_size, n)


# ─── Model ───────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, h, d // h).transpose(1, 2)
        k = k.view(B, S, h, d // h).transpose(1, 2)
        v = v.view(B, S, h, d // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, d)
        return self.proj(out)


def estimate_attention_cost(d_model: int, n_heads: int, batch_size: int,
                            seq_len: int, tau: int = 4) -> int:
    """2P + A for the attention sublayer under the stated accounting model."""
    P = tau * (4 * d_model * d_model + 4 * d_model)
    A = tau * batch_size * seq_len * 6 * d_model \
        + tau * batch_size * n_heads * seq_len * seq_len
    return 2 * P + A


class LMBlock(nn.Module):
    """Attention sublayer + GatedMoE FFN sublayer (router-compatible surface)."""

    def __init__(self, d_model, n_heads, d_ff_shared, d_ff_expert, num_experts,
                 admissible_mask=None):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.shared = SharedBase(d_model, d_ff_shared)
        widths = (d_ff_expert if isinstance(d_ff_expert, (list, tuple))
                  else [d_ff_expert] * num_experts)
        assert len(widths) == num_experts
        self.experts = nn.ModuleList([Expert(d_model, w) for w in widths])
        self.norm2 = nn.LayerNorm(d_model)
        self.num_experts = num_experts
        if admissible_mask is None:
            self.admissible = [True] * num_experts
        else:
            self.admissible = [i in set(admissible_mask) for i in range(num_experts)]

    def forward_ffn(self, x, gate, debt, mc, batch_size, seq_len):
        """Identical decision logic to GatedMoELayer.forward."""
        residual = x
        h = self.norm1(x)
        shared_out = self.shared(h)
        expert_costs = [gate.estimate_expert_cost(e, batch_size, seq_len)
                        for e in self.experts]
        feasible_mask = torch.tensor(
            [adm and mc.can_fit(c) for adm, c in zip(self.admissible, expert_costs)],
            dtype=torch.bool, device=debt.debt.device)
        fsize = int(feasible_mask.sum().item())
        if feasible_mask.any():
            cand = debt.select_expert(feasible_mask)
            mc.load_from_flash(expert_costs[cand])
            out = shared_out + self.experts[cand](h)
            activated = cand
        else:
            out = shared_out
            activated = None
        return self.norm2(residual + out), activated, fsize


class GatedMoELM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff_shared=256,
                 d_ff_expert=None, num_experts=8, num_layers=4, max_seq=128,
                 admissible_mask=None):
        super().__init__()
        if d_ff_expert is None:
            d_ff_expert = [64, 96, 128, 192, 256, 384, 512, 1024][:num_experts]
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq, d_model))
        self.layers = nn.ModuleList([
            LMBlock(d_model, n_heads, d_ff_shared, d_ff_expert, num_experts,
                    admissible_mask) for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.num_layers = num_layers
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def backbone_cost(self, gate, batch_size, seq_len) -> int:
        layer = self.layers[0]
        return (estimate_attention_cost(self.d_model, self.n_heads,
                                        batch_size, seq_len)
                + gate.estimate_shared_cost(layer.shared, batch_size, seq_len))

    def forward(self, x, gate, debt_counters, mc):
        B, S = x.shape
        h = self.embed(x) + self.pos[:, :S]
        activated_experts, feasible_sizes = [], []
        for i, layer in enumerate(self.layers):
            bcost = self.backbone_cost(gate, B, S)
            mc.allocate(bcost)
            h = h + layer.attn(layer.norm_attn(h))
            h, activated, fsize = layer.forward_ffn(
                h, gate, debt_counters[i], mc, B, S)
            activated_experts.append(activated)
            feasible_sizes.append(fsize)
            mc.deallocate(bcost)
            if activated is not None:
                mc.deallocate(gate.estimate_expert_cost(
                    layer.experts[activated], B, S))
        logits = self.head(self.norm_f(h))
        return logits, {"activated_experts": activated_experts,
                        "feasible_sizes": feasible_sizes}
