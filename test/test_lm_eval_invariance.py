"""Amendment A4 (2026-09-30) regression tests for LM evaluation side effects.

A code audit found that (1) LM evaluation never advanced the debt counters of the
debt-based models, so they evaluated with one frozen expert per layer, and (2)
evaluation mutated the schedule state (position, permutation) of the fixed-cadence and
cycle-shuffle routers, which changed their subsequent TRAINING routes.

These tests require, for every LM routing method:
  1. eval_ppl leaves every router's schedule state and every debt counter unchanged;
  2. inserting evaluation passes of different lengths does not change any training route;
  3. uniform-debt evaluation routes equal round-robin evaluation routes (Proposition 1).
A final test shows that the pre-A4 evaluation function fails checks 1-3 (the tests can fail).
"""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from gatedmoe.core_gate_memory import VirtualMemoryCounter, PredictiveMemoryGate
from gatedmoe.core_debt_scheduler import DebtCounter, GradientDebtCounter
from gatedmoe.lm_train_loop import make_lm, eval_ppl

VOCAB, B, BPTT = 50, 4, 8
WIDTHS = [8, 12, 16, 24]
METHODS = ["round_robin", "gatedmoe_base", "gatedmoe_g", "random", "learned_top1", "gated_top1",
           "learned_top1_aux0", "fixed_cadence_random", "cycle_shuffle"]
CFG = dict(seed=7, d_model=16, n_heads=2, d_ff_shared=32, widths=WIDTHS, num_layers=2,
           bptt=BPTT, admissible=None)


def _eval_ppl_pre_a4(model, data, cfg, gate, debt, mc, device):
    """The evaluation function as shipped before amendment A4 (kept to prove the tests fail)."""
    model.eval()
    counters = copy.deepcopy(debt)
    states = []
    if hasattr(model, "routers"):
        states = [(getattr(r, "current", None),
                   r.gen.get_state().clone() if hasattr(r, "gen") else None)
                  for r in model.routers]
    total_loss, total_tok = 0.0, 0
    bptt = cfg["bptt"]
    with torch.no_grad():
        for pos in range(0, ((data.shape[1] - 1) // bptt) * bptt, bptt):
            x = data[:, pos:pos + bptt].to(device)
            y = data[:, pos + 1:pos + 1 + x.shape[1]].to(device)
            mc.reset_step()
            logits, _ = model(x, gate, counters, mc)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tok += y.numel()
    if hasattr(model, "routers"):
        for r, (cur, gen) in zip(model.routers, states):
            if cur is not None:
                r.current = cur
            if gen is not None:
                r.gen.set_state(gen)
    model.train()
    return math.exp(total_loss / total_tok)


def _data(seed, length):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, length), generator=g)


def _setup(method):
    cfg = dict(CFG, method=method)
    model = make_lm(cfg, VOCAB)
    mc = VirtualMemoryCounter(10 ** 12)  # every expert feasible
    gate = PredictiveMemoryGate(mc)
    K = len(WIDTHS)
    debt = [GradientDebtCounter(K) if method == "gatedmoe_g" else DebtCounter(K)
            for _ in range(cfg["num_layers"])]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return cfg, model, mc, gate, debt, opt


def _train_step(model, x, y, gate, debt, mc, opt):
    """Mirrors one step of run_lm_experiment's training loop."""
    mc.reset_step()
    logits, info = model(x, gate, debt, mc)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    if hasattr(model, "routers"):
        for r in model.routers:
            if (hasattr(r, "aux_loss") and isinstance(r.aux_loss, torch.Tensor)
                    and not getattr(r, "aux_off", False)):
                loss = loss + 0.01 * r.aux_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    if not hasattr(model, "routers"):
        lv = loss.item()
        for i, dc in enumerate(debt):
            a = info["activated_experts"][i]
            if isinstance(dc, GradientDebtCounter):
                dc.step(a, current_loss=lv)
            else:
                dc.step(a)
    return [-1 if a is None else a for a in info["activated_experts"]]


def _router_state(model):
    if not hasattr(model, "routers"):
        return None
    out = []
    for r in model.routers:
        st = {k: copy.deepcopy(getattr(r, k)) for k in ("current", "pos", "order") if hasattr(r, k)}
        if hasattr(r, "gen"):
            st["gen"] = r.gen.get_state().clone()
        out.append(st)
    return out


def _debt_state(debt):
    return [{k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
             for k, v in vars(dc).items()} for dc in debt]


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _state_unchanged_by_eval(method, eval_fn):
    cfg, model, mc, gate, debt, opt = _setup(method)
    train = _data(1, BPTT * 6 + 1)
    for s in range(5):
        x = train[:, s * BPTT:(s + 1) * BPTT]
        y = train[:, s * BPTT + 1:(s + 1) * BPTT + 1]
        _train_step(model, x, y, gate, debt, mc, opt)
    before_r, before_d = _router_state(model), _debt_state(debt)
    eval_fn(model, _data(2, BPTT * 5 + 1), cfg, gate, debt, mc, torch.device("cpu"))
    return _same(before_r, _router_state(model)) and _same(before_d, _debt_state(debt))


def _routes(method, eval_fn, insert_at):
    cfg, model, mc, gate, debt, opt = _setup(method)
    train = _data(1, BPTT * 12 + 1)
    routes = []
    for s in range(12):
        if s in insert_at:
            eval_fn(model, _data(10 + s, BPTT * insert_at[s] + 1), cfg, gate, debt, mc,
                    torch.device("cpu"))
        x = train[:, s * BPTT:(s + 1) * BPTT]
        y = train[:, s * BPTT + 1:(s + 1) * BPTT + 1]
        routes.append(_train_step(model, x, y, gate, debt, mc, opt))
    return routes


def _eval_routes(method, eval_fn):
    """Routes chosen during one evaluation pass from a fresh model (captured via a wrapper)."""
    cfg, model, mc, gate, debt, opt = _setup(method)
    seen = []
    fwd = model.forward

    def recording_forward(*a, **k):
        logits, info = fwd(*a, **k)
        seen.append([-1 if e is None else e for e in info["activated_experts"]])
        return logits, info

    model.forward = recording_forward
    eval_fn(model, _data(3, BPTT * 9 + 1), cfg, gate, debt, mc, torch.device("cpu"))
    return seen


def test_eval_leaves_router_and_debt_state_unchanged():
    torch.manual_seed(0)
    bad = [m for m in METHODS if not _state_unchanged_by_eval(m, eval_ppl)]
    assert not bad, f"eval_ppl changed training state for {bad}"


def test_training_routes_invariant_to_inserted_evaluation():
    torch.manual_seed(0)
    for m in METHODS:
        plain = _routes(m, eval_ppl, {})
        with_eval = _routes(m, eval_ppl, {3: 5, 7: 2, 8: 7})
        assert plain == with_eval, f"{m}: evaluation changed later training routes"


def test_uniform_debt_evaluation_routes_equal_round_robin():
    rr = _eval_routes("round_robin", eval_ppl)
    db = _eval_routes("gatedmoe_base", eval_ppl)
    assert rr == db, f"round-robin {rr} vs debt {db}"
    assert len({tuple(r) for r in rr}) > 1, "evaluation routes do not cycle"


def test_pre_a4_evaluation_fails_these_checks():
    """Proof that the checks can fail: the pre-A4 function violates each property."""
    torch.manual_seed(0)
    assert not _state_unchanged_by_eval("fixed_cadence_random", _eval_ppl_pre_a4)
    assert not _state_unchanged_by_eval("cycle_shuffle", _eval_ppl_pre_a4)
    assert (_routes("fixed_cadence_random", _eval_ppl_pre_a4, {})
            != _routes("fixed_cadence_random", _eval_ppl_pre_a4, {3: 5, 7: 2, 8: 7}))
    assert _eval_routes("round_robin", _eval_ppl_pre_a4) != _eval_routes("gatedmoe_base", _eval_ppl_pre_a4)
