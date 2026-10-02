"""E6-LM training loop (mirrors core_train_loop's protocol guarantees).

Endpoints (PROTOCOL.md): PRIMARY = final-epoch TEST perplexity; SECONDARY =
test ppl at best-VALIDATION-ppl epoch. WikiText-2's canonical valid/test
splits serve as validation/test. Contiguous batching => batch order is
matched across methods by construction; RandomRouter keeps its isolated RNG.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter, GradientDebtCounter
from gatedmoe.lm_model_define import GatedMoELM, load_wikitext2, batchify
from gatedmoe.sim_baseline_routers import (LearnedTop1Router, GatedTop1Router,
                                           RandomRouter, RoundRobinRouter,
                                           FixedCadenceRandomRouter,
                                           CycleShuffleRouter)


class BaselineLM(nn.Module):
    """Router-driven variant of GatedMoELM (batch-level routers reuse LMBlock)."""

    def __init__(self, model: GatedMoELM, routers, method: str):
        super().__init__()
        self.model = model
        self.routers = nn.ModuleList(
            routers) if isinstance(routers[0], nn.Module) else routers
        self.method = method

    def forward(self, x, gate, debt_counters, mc):
        B, S = x.shape
        m = self.model
        h = m.embed(x) + m.pos[:, :S]
        activated_experts, feasible_sizes = [], []
        for i, layer in enumerate(m.layers):
            bcost = m.backbone_cost(gate, B, S)
            mc.allocate(bcost)
            h = h + layer.attn(layer.norm_attn(h))
            residual = h
            h_norm = layer.norm1(h)
            fsize = sum(1 for j, e in enumerate(layer.experts)
                        if layer.admissible[j]
                        and mc.can_fit(gate.estimate_expert_cost(e, B, S)))
            feasible_sizes.append(fsize)
            router = self.routers[i]
            out, activated = router.route(layer, h_norm, gate, mc, B, S)
            h = layer.norm2(residual + out)
            activated_experts.append(activated)
            mc.deallocate(bcost)
            if activated is not None and not isinstance(activated, list):
                mc.deallocate(gate.estimate_expert_cost(
                    layer.experts[activated], B, S))
            debt_counters[i].step(
                activated if not isinstance(activated, list) else None)
        logits = m.head(m.norm_f(h))
        return logits, {"activated_experts": activated_experts,
                        "feasible_sizes": feasible_sizes}

    def parameters(self, recurse=True):
        yield from self.model.parameters(recurse=recurse)
        for r in self.routers:
            if isinstance(r, nn.Module):
                yield from r.parameters(recurse=recurse)


def make_lm(cfg, vocab_size):
    torch.manual_seed(cfg["seed"])
    base = GatedMoELM(vocab_size, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                      d_ff_shared=cfg["d_ff_shared"], d_ff_expert=cfg["widths"],
                      num_experts=len(cfg["widths"]), num_layers=cfg["num_layers"],
                      max_seq=cfg["bptt"], admissible_mask=cfg.get("admissible"))
    method = cfg["method"]
    n_layers, d = cfg["num_layers"], cfg["d_model"]
    K = len(cfg["widths"])
    if method in ("gatedmoe_base", "gatedmoe_g"):
        return base
    if method == "round_robin":
        routers = [RoundRobinRouter(K) for _ in range(n_layers)]
    elif method == "random":
        routers = [RandomRouter(K, seed=2000 + cfg["seed"] + 17 * i)
                   for i in range(n_layers)]
    elif method == "learned_top1":
        routers = [LearnedTop1Router(d, K) for _ in range(n_layers)]
    elif method == "gated_top1":
        routers = [GatedTop1Router(d, K) for _ in range(n_layers)]
    elif method == "fixed_cadence_random":
        routers = [FixedCadenceRandomRouter(K, seed=3000 + cfg["seed"] + 17 * i)
                   for i in range(n_layers)]
    elif method == "cycle_shuffle":
        routers = [CycleShuffleRouter(K, seed=4000 + cfg["seed"] + 17 * i)
                   for i in range(n_layers)]
    elif method == "learned_top1_aux0":
        routers = [LearnedTop1Router(d, K) for _ in range(n_layers)]
        for r in routers:
            r.aux_off = True  # A3.1: auxiliary loss disabled
    else:
        raise ValueError(method)
    return BaselineLM(base, routers, method)


@torch.no_grad()
def eval_ppl(model, data, cfg, gate, debt, mc, device):
    import copy
    model.eval()
    counters = copy.deepcopy(debt)
    # Protocol amendment A4 (2026-09-30, defect 2): snapshot the COMPLETE
    # schedule state of every router (cyclic position, permutation position and order,
    # RNG), so that evaluation can never change the training-time schedule.
    states = []
    if hasattr(model, "routers"):
        for r in model.routers:
            st = {k: copy.deepcopy(getattr(r, k))
                  for k in ("current", "pos", "order") if hasattr(r, k)}
            st["_gen"] = r.gen.get_state().clone() if hasattr(r, "gen") else None
            states.append(st)
    total_loss, total_tok = 0.0, 0
    bptt = cfg["bptt"]
    # Full windows only: a ragged final window shrinks activation costs and
    # momentarily enlarges |F|, desynchronizing debt vs cyclic scheduling
    # (protocol amendment 2026-08-22).
    for pos in range(0, ((data.shape[1] - 1) // bptt) * bptt, bptt):
        x = data[:, pos:pos + bptt].to(device)
        y = data[:, pos + 1:pos + 1 + x.shape[1]].to(device)
        if y.shape[1] < x.shape[1]:
            x = x[:, :y.shape[1]]
        mc.reset_step()
        logits, info = model(x, gate, counters, mc)
        if not hasattr(model, "routers"):
            # Amendment A4 (defect 1): debt-based models route from the
            # counters, so advance the evaluation copy after every window, exactly as
            # vision evaluation does (core_train_loop.evaluate).
            for i, dc in enumerate(counters):
                dc.step(info["activated_experts"][i])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()
    if hasattr(model, "routers"):
        for r, st in zip(model.routers, states):
            for k, v in st.items():
                if k == "_gen":
                    if v is not None:
                        r.gen.set_state(v)
                else:
                    setattr(r, k, v)
    model.train()
    return math.exp(total_loss / total_tok)


def run_lm_experiment(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, valid_ids, test_ids, vocab_size = load_wikitext2(cfg["data_dir"])
    train = batchify(train_ids, cfg["batch_size"])
    valid = batchify(valid_ids, cfg["batch_size"])
    test = batchify(test_ids, cfg["batch_size"])

    model = make_lm(cfg, vocab_size).to(device)
    mc = VirtualMemoryCounter(cfg["budget"])
    gate = PredictiveMemoryGate(mc)
    K = len(cfg["widths"])
    debt = [GradientDebtCounter(K) if cfg["method"] == "gatedmoe_g"
            else DebtCounter(K) for _ in range(cfg["num_layers"])]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    results = {"epochs": [], "config": dict(cfg, vocab_size=vocab_size,
                                            protocol="s0-lm-a4")}
    route_trace = []
    best_val, best_val_epoch, test_at_best_val = float("inf"), -1, None
    bptt = cfg["bptt"]
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        feasible_hist = [0] * (K + 1)
        total_loss, total_task, total_tok = 0.0, 0.0, 0
        # Full windows only (see eval_ppl note / protocol amendment 2026-08-22)
        for pos in range(0, ((train.shape[1] - 1) // bptt) * bptt, bptt):
            x = train[:, pos:pos + bptt].to(device)
            y = train[:, pos + 1:pos + 1 + x.shape[1]].to(device)
            if y.shape[1] < x.shape[1]:
                x = x[:, :y.shape[1]]
            if x.shape[1] < 2:
                continue
            mc.reset_step()
            logits, info = model(x, gate, debt, mc)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            task_loss_val = loss.item()  # A4: task CE, logged separately
            if hasattr(model, "routers"):
                for r in model.routers:
                    if (hasattr(r, "aux_loss") and isinstance(r.aux_loss, torch.Tensor)
                            and not getattr(r, "aux_off", False)):
                        loss = loss + 0.01 * r.aux_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not hasattr(model, "routers"):
                lv = loss.item()
                for i, dc in enumerate(debt):
                    a = info["activated_experts"][i]
                    if isinstance(dc, GradientDebtCounter):
                        dc.step(a, current_loss=lv)
                    else:
                        dc.step(a)
            for f in info["feasible_sizes"]:
                feasible_hist[min(f, K)] += 1
            route_trace.append([-1 if a is None else a
                                for a in info["activated_experts"]])
            total_loss += loss.item() * y.numel()
            total_task += task_loss_val * y.numel()
            total_tok += y.numel()
        val_ppl = eval_ppl(model, valid, cfg, gate, debt, mc, device)
        test_ppl = eval_ppl(model, test, cfg, gate, debt, mc, device)
        sched.step()
        ep = {
            # A4: train_ppl = task cross-entropy only (all arms); train_obj_ppl = the
            # optimized objective including any balancing penalty (the pre-A4 train_ppl).
            "epoch": epoch, "train_ppl": math.exp(total_task / total_tok),
            "train_obj_ppl": math.exp(total_loss / total_tok),
            "val_ppl": val_ppl, "test_ppl": test_ppl,
            "peak_sram_bytes": mc.peak_bytes,
            "flash_to_sram_bytes": mc.flash_to_sram_bytes,
            "feasible_hist": feasible_hist,
            "time_s": time.time() - t0,
            "debt_stats": [dc.get_stats() for dc in debt],
        }
        results["epochs"].append(ep)
        if val_ppl < best_val:
            best_val, best_val_epoch, test_at_best_val = val_ppl, epoch, test_ppl
        print(f"[{cfg['name']}] Epoch {epoch + 1:2d} | train_ppl "
              f"{ep['train_ppl']:.1f} | val {val_ppl:.1f} | test {test_ppl:.1f} "
              f"| peak {mc.peak_bytes / 1024:.0f}KB | {ep['time_s']:.0f}s",
              flush=True)

    results["final_test_ppl"] = results["epochs"][-1]["test_ppl"]  # PRIMARY
    results["best_val_ppl"] = best_val
    results["best_val_epoch"] = best_val_epoch
    results["test_ppl_at_best_val"] = test_at_best_val  # SECONDARY

    out = Path(cfg["save_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "routes_train.npy", np.asarray(route_trace, dtype=np.int8))
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
