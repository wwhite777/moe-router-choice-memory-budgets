"""Recompute the manuscript numbers that no other analysis script produces.

Reads only per-run files (result/logs/<run>/results.json, routes_train.npy) and
the tensor-memory measurement JSON; writes result/tables/numbers_v11.csv with
columns id, quantity, value, definition. Refuses to overwrite an existing output.
Its run-inventory rows (inv_*) cover the runs before Amendment 5; the full
inventory, including the e6lm5_, e1a5_ and e1a5chk_ runs, is scripts/count-runs-v14.py.

Changes against the previous analysis (2026-09-30, PROTOCOL.md Amendment 4):
  * The LM evidence is the corrected rerun: e6lm3_* (d=128, 200 runs) and
    e6lm3d256_* (d=256, 40 runs). The older e6lm2_* and e6lmd256_* runs are
    SUPERSEDED (debt counters were not advanced during LM evaluation, and
    evaluation advanced the position of fixed_cadence_random / cycle_shuffle
    routers). They are counted separately in the inventory and compared with the
    corrected runs in section 11; they feed no paper number.
  * Inventory: inv_total_paper (superseded prefixes excluded), inv_superseded,
    inv_total_all; failure scan and infrastructure failures over the paper runs,
    and separately over the superseded runs.
  * Section 6 adds the 7-arm gap-CV correlations (gatedmoe_base excluded), the
    three d=256 contrasts vs round_robin at the final and best-validation
    endpoints (raw and Holm-over-three p), pooled |F|=4 gap histograms, and a
    TOST of fixed_cadence_random vs round_robin at |F|=8.
  * Section 11 (rerun comparison) pairs e6lm2<->e6lm3 and e6lmd256<->e6lm3d256
    by (method, d_model, budget, seed). It always compares the fixed
    SUPERSEDED grids with the fixed CORRECTED grids, whatever the flags below.
  * Trace fallback (from the earlier supplement copy): runs without
    routes_train.npy (the archive omits it for e1_/e4_) use the frozen
    route_disagreement.csv for disagreement and route_class_counts.npy for the
    experts used, so the same file runs in the repository and in the archive.

Usage (from the repository root):
  python3 scripts/reproduce_numbers_v11.py [--out PATH] [--lm-main PREFIX] [--lm-wide PREFIX]
    --lm-main  LM grid for d=128 numbers (sections 6, 9, 10); default e6lm3
    --lm-wide  LM grid for d=256 numbers (sections 6, 9);      default e6lm3d256
  e.g. --lm-main e6lm2 --lm-wide e6lmd256 --out numbers_superseded.csv reproduces the
  superseded values for comparison. A grid with no completed runs is an error.
Dependencies: numpy, scipy (standard library otherwise).
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "result" / "logs"
FROZEN = ROOT / "result" / "frozen_2026-08-24"
E3_JSON = ROOT / "result" / "e3_measured_memory.json"
OUT_DEFAULT = ROOT / "result" / "tables" / "numbers_v11.csv"
# Used only for runs whose routes_train.npy is absent (the e1_/e4_ traces are
# not shipped): per-seed route disagreement from the frozen table (4 decimals;
# on the full traces no nonzero disagreement is below 5e-5, the smallest is
# 0.498), and the experts used from route_class_counts.npy (same training
# steps as the trace; slot 0 = null path).
ROUTE_TABLE = FROZEN / "route_disagreement.csv"

ARMS = ["s1", "e1", "e4", "e5", "e5tune", "e6lm", "e6lm2", "e6lmd256", "e6lm3", "e6lm3d256"]
PREFIX_ORDER = ["e6lm3d256", "e6lmd256", "e6lm3", "e6lm2", "e6lm", "e5tune", "e5", "e4", "e1", "s1"]  # longest first
LM_ARMS = ("e6lm", "e6lm2", "e6lmd256", "e6lm3", "e6lm3d256")
LM_MAIN = "e6lm3"          # corrected d=128 grid (Amendment 4)
LM_WIDE = "e6lm3d256"      # corrected d=256 grid (Amendment 4)
SUPERSEDED = ("e6lm2", "e6lmd256")
CORRECTED = ("e6lm3", "e6lm3d256")  # section 11 partners of SUPERSEDED, position by position
VIS_PAT = re.compile(r"^(s1|e1|e4|e5)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")
LM_PAT = re.compile(r"^(e6lm3d256|e6lmd256|e6lm3|e6lm2|e6lm)_(\w+?)_B(\d+)k_s(\d+)$")
LM_F = {17991: 0, 24457: 4, 42487: 8}  # LM budget (KiB) -> realized |F| (checked below)
PATCH = {"s1": 4, "e1": 4, "e4": 8, "e5": 4}  # patch size on 32x32 inputs -> S = (32/p)^2 tokens
TAU = 4  # bytes per fp32 value
RR, DB = "round_robin", "gatedmoe_base"
DG = "gatedmoe_g"
FCR = "fixed_cadence_random"

ROWS = []


def add(qid, quantity, value, definition):
    ROWS.append({"id": qid, "quantity": quantity, "value": value, "definition": definition})


def fmt(x, nd=4):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def fmt_p(p):
    """p-values: six significant digits (keeps very small values readable)."""
    return f"{float(p):.6g}"


# ── statistics ────────────────────────────────────────────────────────────────
def paired_p(a, b):
    """Two-sided paired t-test p; identical vectors -> 1.0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0) or d.std(ddof=1) == 0:
        return 1.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def holm(p):
    p = np.asarray(p, float)
    adj, cur = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p, kind="mergesort")):
        cur = max(cur, (len(p) - k) * p[i])
        adj[i] = min(1.0, cur)
    return adj


def paired_power(delta, sd_diff, n, alpha=0.05):
    df, ncp = n - 1, delta / (sd_diff / math.sqrt(n))
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))


def tost_paired(d, margin):
    """Two one-sided paired t-tests of |mean diff| < margin; returns p_tost and the 90% CI."""
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    p_lower = 1 - stats.t.cdf((m + margin) / se, df)  # H0: diff <= -margin
    p_upper = stats.t.cdf((m - margin) / se, df)      # H0: diff >= +margin
    h = stats.t.ppf(0.95, df) * se
    return float(max(p_lower, p_upper)), float(m - h), float(m + h), float(p_lower), float(p_upper)


def chat(b, S, d, f, tau=TAU):
    """Closed-form training cost of a Linear-ReLU-Linear block: 2P + A (1-byte ReLU mask)."""
    P = tau * (2 * d * f + f + d)
    A = tau * b * S * (2 * d + 2 * f) + b * S * f
    return 2 * P + A


def delta_ln(b, S, d, tau=TAU):
    return 2 * tau * b * S * d + 4 * tau * b * S


def gap_cv(routes):
    """Mean over (layer, expert with >2 activations) of std(gaps, ddof=0)/mean(gaps)."""
    cvs = []
    for l in range(routes.shape[1]):
        for e in range(8):
            idx = np.where(routes[:, l] == e)[0]
            if len(idx) > 2:
                g = np.diff(idx)
                cvs.append(g.std() / g.mean())
    return float(np.mean(cvs))


def gap_hist(routes):
    """Counts of inter-activation gaps (steps) over all layers and non-null experts."""
    vals = Counter()
    for l in range(routes.shape[1]):
        col = routes[:, l]
        for e in np.unique(col[col >= 0]):
            vals.update(np.diff(np.where(col == e)[0]).tolist())
    return vals


# ── loading ───────────────────────────────────────────────────────────────────
def prefix_of(name):
    return next((p for p in PREFIX_ORDER if name.startswith(p + "_")), None)


def load_runs():
    runs = []
    for d in sorted(LOGS.iterdir()):
        p = prefix_of(d.name)
        if p is None or not d.is_dir():
            continue
        r = {"name": d.name, "prefix": p, "dir": d, "ok": (d / "results.json").exists()}
        if r["ok"]:
            j = json.load(open(d / "results.json"))
            cfg, eps = j.get("config", {}), j.get("epochs", [])
            r.update(cfg=cfg, n_ep=len(eps), peaks=[e.get("peak_sram_bytes") for e in eps],
                     fhist=[e.get("feasible_hist") for e in eps])
            if p in LM_ARMS:
                g, meth, bkb, seed = LM_PAT.match(d.name).groups()
                r.update(kind="lm", method=meth, budget_kb=int(bkb), seed=int(seed),
                         final=j.get("final_test_ppl"), at_best_val=j.get("test_ppl_at_best_val"),
                         curve=[e.get("test_ppl") for e in eps])
            else:
                if p == "e5tune":
                    ds, cf, s = d.name.split("_")[1:4]
                    r.update(grid="e5tune", dataset=ds, method="token_capacity", cell=cf, seed=int(s[1:]))
                else:
                    g, ds, meth, cell, seed = VIS_PAT.match(d.name).groups()
                    r.update(grid=g, dataset=ds, method=meth, cell=cell, seed=int(seed))
                r.update(kind="vision", final=j.get("final_test_acc"), loss=j.get("final_test_loss"),
                         best_val_acc=j.get("best_val_acc"))
        runs.append(r)
    return runs


def realized_F(r):
    fh = np.array(r["fhist"]).sum(axis=0)
    return sorted(int(i) for i in np.nonzero(fh)[0])


def infra_failures(runs):
    return sum(1 for r in runs if not r["ok"]) + sum(
        1 for r in runs if r["ok"] and (r["n_ep"] != r["cfg"].get("epochs") or r["final"] is None
                                         or not math.isfinite(r["final"])))


def failure_scan(ok_runs):
    groups = defaultdict(list)
    for r in ok_runs:
        groups[r["name"].rsplit("_s", 1)[0]].append(r)
    hits = []
    for key, rs in groups.items():
        for r in rs:
            sib = [x["final"] for x in rs if x is not r]
            if not sib:
                continue
            med = float(np.median(sib))
            if (r["kind"] == "vision" and r["final"] < med - 0.10) or (
                    r["kind"] == "lm" and (not math.isfinite(r["final"]) or r["final"] > 2 * med)):
                hits.append((r, sib))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--lm-main", default=LM_MAIN, help="LM grid prefix for the d=128 numbers")
    ap.add_argument("--lm-wide", default=LM_WIDE, help="LM grid prefix for the d=256 numbers")
    args = ap.parse_args()
    out = Path(args.out)
    lm_main, lm_wide = args.lm_main, args.lm_wide
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")
    for flag, v in (("--lm-main", lm_main), ("--lm-wide", lm_wide)):
        if v not in LM_ARMS:
            sys.exit(f"{flag} {v!r} is not an LM prefix (choose from {', '.join(LM_ARMS)})")

    runs = load_runs()
    ok = [r for r in runs if r["ok"]]
    for flag, v in (("--lm-main", lm_main), ("--lm-wide", lm_wide)):
        if not any(r["prefix"] == v for r in ok):
            sys.exit(f"{flag} {v}: no completed runs under {LOGS}")
    routes_cache = {}

    def routes(r):
        if r["name"] not in routes_cache:
            routes_cache[r["name"]] = np.load(r["dir"] / "routes_train.npy")
        return routes_cache[r["name"]]

    def has_trace(r):
        return (r["dir"] / "routes_train.npy").exists()

    def used_experts(r):
        if has_trace(r):
            return [int(x) for x in np.unique(routes(r)) if x >= 0]
        rcc = np.load(r["dir"] / "route_class_counts.npy")  # [L, K+1, C]
        return [e for e in range(rcc.shape[1] - 1) if rcc[:, e + 1].sum() > 0]

    frozen_dis = {}

    def disagreement(k, s, a, b):
        ra, rb = acc[k][(a, s)], acc[k][(b, s)]
        if has_trace(ra) and has_trace(rb):
            return float((routes(ra) != routes(rb)).mean())
        if not frozen_dis:
            with open(ROUTE_TABLE) as fh:
                for x in csv.DictReader(fh):
                    frozen_dis[(x["grid"], x["dataset"], x["cell"], int(x["seed"]),
                                frozenset((x["method_a"], x["method_b"])))] = float(x["disagreement"])
        return frozen_dis[(k[0], k[1], k[2], s, frozenset((a, b)))]

    # ── 1 inventory ──────────────────────────────────────────────────────────
    cnt = Counter(r["prefix"] for r in runs)
    for a in ARMS:
        add(f"inv_{a}", f"training runs, arm {a}" + (" (superseded)" if a in SUPERSEDED else ""), cnt[a],
            f"run directories named {a}_*")
    paper_prefixes = [a for a in ARMS if a not in SUPERSEDED]
    n_paper = sum(cnt[a] for a in paper_prefixes)
    n_sup = sum(cnt[a] for a in SUPERSEDED)
    add("inv_total_paper", "training runs, paper arms", n_paper,
        f"sum over {', '.join(paper_prefixes)} (superseded {', '.join(SUPERSEDED)} excluded)")
    add("inv_superseded", "training runs, superseded LM arms", n_sup,
        f"sum over {', '.join(SUPERSEDED)} (Amendment 4: LM evaluation defect)")
    add("inv_total_all", "training runs, all arms on disk", n_paper + n_sup, "inv_total_paper + inv_superseded")
    add("inv_batch_level", "batch-level paper runs (paper arms except e5, e5tune)",
        n_paper - cnt["e5"] - cnt["e5tune"], "inv_total_paper minus token-level arms")
    paper_runs = [r for r in runs if r["prefix"] not in SUPERSEDED]
    sup_runs = [r for r in runs if r["prefix"] in SUPERSEDED]
    add("inv_infra_failures", "infrastructure failures, paper runs", infra_failures(paper_runs),
        "runs without results.json, with fewer epochs than configured, or with a non-finite primary endpoint")
    add("inv_infra_failures_superseded", "infrastructure failures, superseded runs", infra_failures(sup_runs), "same")
    hits = failure_scan([r for r in paper_runs if r["ok"]])
    add("fail_scan_hits", "training-failure scan: number of hits, paper runs", len(hits),
        "vision: final test acc < median of sibling seeds (same config) - 0.10; LM: non-finite or > 2x sibling median ppl")
    add("fail_scan_runs", "training-failure scan: runs hit, paper runs", ";".join(h[0]["name"] for h in hits), "same scan")
    if len(hits) == 1:
        r, sib = hits[0]
        add("fail_final_acc", "failed run final test accuracy", fmt(r["final"]), "results.json final_test_acc")
        add("fail_sibling_min", "failed run: sibling seeds final accuracy, min", fmt(min(sib)), "other seeds, same config")
        add("fail_sibling_max", "failed run: sibling seeds final accuracy, max", fmt(max(sib)), "other seeds, same config")
    hits_sup = failure_scan([r for r in sup_runs if r["ok"]])
    add("fail_scan_hits_superseded", "training-failure scan: number of hits, superseded runs", len(hits_sup), "same scan")
    add("fail_scan_runs_superseded", "training-failure scan: runs hit, superseded runs",
        ";".join(h[0]["name"] for h in hits_sup), "same scan")

    # ── 2 gated vision runs ─────────────────────────────────────────────────────
    gated = [r for r in ok if r["prefix"] in ("s1", "e1", "e4")]
    n_budget_ok = n_eq = 0
    e4F = defaultdict(set)
    for r in gated:
        c = r["cfg"]
        S = (32 // PATCH[r["prefix"]]) ** 2
        widths = c["d_ff_expert"] if isinstance(c["d_ff_expert"], list) else [c["d_ff_expert"]] * c["num_experts"]
        used = used_experts(r)
        analytic = chat(c["batch_size"], S, c["d_model"], c["d_ff_shared"]) + \
            (max(chat(c["batch_size"], S, c["d_model"], widths[u]) for u in used) if used else 0)
        peak = max(r["peaks"])
        n_budget_ok += peak <= c["budget_bytes"]
        n_eq += peak == analytic
        if r["prefix"] == "e4":
            e4F[c["budget_bytes"] // 1024].update(realized_F(r))
    add("gated_runs", "gated vision runs (s1+e1+e4)", len(gated), "count")
    add("gated_peak_le_budget", "gated runs with virtual peak <= budget", n_budget_ok,
        "max over epochs of peak_sram_bytes <= config budget_bytes")
    add("gated_peak_eq_analytic", "gated runs with virtual peak == analytic sum", n_eq,
        "analytic = C(shared) + max C(expert) over experts in the training route trace, C = 2P + A at the training batch size")
    for bk in sorted(e4F):
        add(f"e4_F_{bk}KiB", f"e4 realized |F| at {bk} KiB", "/".join(map(str, sorted(e4F[bk]))),
            "union of nonzero feasible-set histogram bins over all epochs and runs")

    # ── vision accuracy tables ──────────────────────────────────────────────────
    acc = defaultdict(dict)
    for r in gated:
        acc[(r["grid"], r["dataset"], r["cell"])][(r["method"], r["seed"])] = r
    val = lambda cell, m, s: acc[cell][(m, s)]["final"]

    # ── 3 S1 ─────────────────────────────────────────────────────────────────────
    s1cells = sorted(k for k in acc if k[0] == "s1")
    s1m = sorted({m for k in s1cells for m, _ in acc[k]})
    spread = {}
    for k in s1cells:
        seeds = sorted({s for _, s in acc[k]})
        spread[k[2]] = np.mean([max(val(k, m, s) for m in s1m) - min(val(k, m, s) for m in s1m) for s in seeds])
    ge2 = [c for c in spread if not c.startswith("m1v")]
    add("s1_router_range_min", "S1 per-seed router range, min over m>=2 cells (pp)", fmt(min(spread[c] for c in ge2) * 100, 3),
        "per cell: mean over seeds of (max - min final acc across the 4 routers)")
    add("s1_router_range_max", "S1 per-seed router range, max over m>=2 cells (pp)", fmt(max(spread[c] for c in ge2) * 100, 3), "same")
    for ds in ("c100", "c10"):
        k = next(k for k in acc if k[0] == "e1" and k[1] == ds
                 and realized_F(next(iter(acc[k].values()))) == [0])
        ms = sorted({m for m, _ in acc[k]})
        seeds = sorted({s for _, s in acc[k]})
        fl = np.mean([max(val(k, m, s) for m in ms) - min(val(k, m, s) for m in ms) for s in seeds])
        add(f"floor_{ds}", f"|F|=0 no-routing floor, {ds} (pp)", fmt(fl * 100, 3),
            "e1 cell with realized |F|=0: mean over seeds of (max - min final acc across the 5 routers)")
    for m in (2, 3, 4, 6):
        k0, k1 = ("s1", "c100", f"m{m}v0"), ("s1", "c100", f"m{m}v1")
        seeds = sorted({s for _, s in acc[k0]} & {s for _, s in acc[k1]})
        se = np.mean([abs(val(k0, mt, s) - val(k1, mt, s)) for mt in s1m for s in seeds])
        add(f"s1_subset_m{m}", f"S1 subset effect at m={m} (pp)", fmt(se * 100, 3),
            "mean over routers and seeds of |acc(subset v0) - acc(subset v1)|, same router and seed")
    ident = tot = 0
    for k in [k for k in s1cells if k[2].startswith("m1v")]:
        for s in sorted({s for _, s in acc[k]}):
            tot += 1
            ident += len({(acc[k][(m, s)]["final"], acc[k][(m, s)]["loss"]) for m in s1m}) == 1
    add("s1_m1_identical", "S1 m=1 (cell, seed) groups with bit-identical final acc and loss across routers",
        f"{ident}/{tot}", "exact equality over the 4 routers")

    # ── 5 route disagreement (needed for 4) ─────────────────────────────────────
    pairs_rows = []
    for k in sorted(acc):
        seeds = sorted({s for _, s in acc[k]})
        ms = sorted({m for m, _ in acc[k]})
        for s in seeds:
            for a, b in combinations(ms, 2):
                if (a, s) not in acc[k] or (b, s) not in acc[k]:
                    continue
                pairs_rows.append({"cell": k, "seed": s, "a": a, "b": b, "dis": disagreement(k, s, a, b),
                                   "dacc": val(k, a, s) - val(k, b, s)})
    tie = [p for p in pairs_rows if {p["a"], p["b"]} == {RR, DB}]
    gen = [p for p in pairs_rows if {p["a"], p["b"]} != {RR, DB} and p["dis"] > 0]
    d = np.array([p["dis"] for p in gen]) * 100
    add("dis_pairs_all", "matched (cell, seed) trace pairs, all router pairs", len(pairs_rows), "s1+e1+e4")
    add("dis_genuine_n", "genuine trace pairs", len(gen), "disagreement > 0, excluding round_robin vs gatedmoe_base")
    add("dis_genuine_min", "disagreement, min over genuine pairs (%)", fmt(d.min(), 2),
        "fraction of (step, layer) decisions with a different expert id (null path counts as a decision)")
    add("dis_genuine_max", "disagreement, max over genuine pairs (%)", fmt(d.max(), 2), "same")
    add("dis_genuine_mean", "disagreement, mean over genuine pairs (%)", fmt(d.mean(), 2), "same")
    add("dis_genuine_absacc", "mean |final-acc difference| over genuine pairs (pp)",
        fmt(np.mean([abs(p["dacc"]) for p in gen]) * 100, 3), "matched seed-level pairs")
    add("rrdb_vision_identical", "round_robin vs gatedmoe_base vision trace pairs identical",
        f"{sum(p['dis'] == 0 for p in tie)}/{len(tie)}", "exact equality of routes_train.npy")

    # ── 4 variance and power ─────────────────────────────────────────────────────
    sds = [np.std([x["final"] for (m, s), x in acc[k].items() if m == meth], ddof=1)
           for k in acc for meth in {m for m, _ in acc[k]}]
    add("sd_run_median", "median per-run SD of final accuracy (pp)", fmt(np.median(sds) * 100, 4),
        "median over (grid, dataset, cell, router) of the across-seed SD (ddof=1), s1+e1+e4")
    tied = defaultdict(list)
    for p in pairs_rows:
        tied[(p["cell"], p["a"], p["b"])].append(p["dis"] == 0)
    sd_gen = []
    for (k, a, b), flags in tied.items():
        if all(flags):
            continue
        seeds = sorted({s for m, s in acc[k] if m == a} & {s for m, s in acc[k] if m == b})
        sd_gen.append(np.std([val(k, a, s) - val(k, b, s) for s in seeds], ddof=1))
    med_pd = float(np.median(sd_gen)) * 100
    add("sd_pairdiff_median_genuine", "median SD of paired differences over genuine cell-pairs (pp)", fmt(med_pd, 4),
        f"cell-pairs whose routes differ in at least one seed (n={len(sd_gen)}); SD over seeds, ddof=1")
    add("power_0p4pp", "power to detect a 0.4 pp paired difference at n=10", fmt(paired_power(0.4, med_pd, 10), 4),
        "two-sided paired t-test, alpha=0.05, noncentral t, sd = sd_pairdiff_median_genuine")

    # ── 6 language model ─────────────────────────────────────────────────────────
    lm = [r for r in ok if r["kind"] == "lm"]
    T = defaultdict(dict)
    for r in lm:
        if r["prefix"] == lm_main:
            F = LM_F[r["budget_kb"]]
            assert realized_F(r) == [F], r["name"]
            T[(r["method"], F)][r["seed"]] = r
    arms8 = sorted(m for m, F in T if F == 8)
    if not arms8:
        sys.exit(f"--lm-main {lm_main}: no |F|=8 runs")
    cvs = {m: np.mean([gap_cv(routes(r)) for r in T[(m, 8)].values()]) for m in arms8}
    ppl = {m: np.mean([r["final"] for r in T[(m, 8)].values()]) for m in arms8}
    x, y = [cvs[m] for m in arms8], [ppl[m] for m in arms8]
    src = f"LM grid {lm_main}_*"
    add("lm_cv_pearson", "Pearson r, arm mean ppl vs arm gap-CV, |F|=8", fmt(stats.pearsonr(x, y)[0], 4),
        f"{src}; over {len(arms8)} arms; gap-CV per run = mean over (layer, expert) of std/mean of inter-activation gaps (ddof=0), averaged over seeds")
    add("lm_cv_spearman", "Spearman rho, arm mean ppl vs arm gap-CV, |F|=8", fmt(stats.spearmanr(x, y)[0], 4),
        "same arms, exact arm means, ties averaged")
    arms7 = [m for m in arms8 if m != DB]
    x7, y7 = [cvs[m] for m in arms7], [ppl[m] for m in arms7]
    add("lm_cv_pearson_7arms", "Pearson r, arm mean ppl vs arm gap-CV, |F|=8, gatedmoe_base excluded",
        fmt(stats.pearsonr(x7, y7)[0], 4), f"{src}; over {len(arms7)} arms (gatedmoe_base duplicates round_robin's trace)")
    add("lm_cv_spearman_7arms", "Spearman rho, arm mean ppl vs arm gap-CV, |F|=8, gatedmoe_base excluded",
        fmt(stats.spearmanr(x7, y7)[0], 4), "same 7 arms, ties averaged")
    for m in arms8:
        add(f"lm_gapcv_F8_{m}", f"gap-CV at |F|=8, {m}", fmt(cvs[m], 3), "as in lm_cv_pearson")
        ents = []
        for r in T[(m, 8)].values():
            rt = routes(r)
            for l in range(rt.shape[1]):
                c = np.bincount(rt[:, l][rt[:, l] >= 0], minlength=8).astype(float)
                q = c[c > 0] / c.sum()
                ents.append(float(-(q * np.log(q)).sum()))
        add(f"lm_entropy_min_{m}", f"min per-layer utilization entropy at |F|=8, {m} (nats)", fmt(min(ents), 4),
            "entropy of per-layer expert activation counts over the training trace; min over seeds and layers (max ln 8)")
        mc = np.mean([r["curve"] for r in T[(m, 8)].values()], axis=0)
        add(f"lm_curve_min_{m}", f"seed-mean test-ppl curve minimum at |F|=8, {m}", fmt(mc.min(), 2), "mean over seeds per epoch")
        add(f"lm_curve_argmin_{m}", f"epoch of that minimum, {m} (1-indexed)", int(np.argmin(mc)) + 1, "same")

    def contrast(A, B):
        s = sorted(set(A) & set(B))
        a = np.array([A[i] for i in s])
        b = np.array([B[i] for i in s])
        return (a - b).mean(), int((a - b > 0).sum()), paired_p(a, b)

    fin = lambda m, F: {s: r["final"] for s, r in T[(m, F)].items()}
    cons = [(ref, contrast(fin(FCR, 8), fin(ref, 8))) for ref in (RR, "random", "learned_top1")]
    adj = holm([c[2] for _, c in cons])
    for (ref, (md, pos, p)), pa in zip(cons, adj):
        add(f"lm_fcr8_vs_{ref}_diff", f"|F|=8 fixed-cadence minus {ref} (ppl)", fmt(md, 2), "mean paired difference over seeds")
        add(f"lm_fcr8_vs_{ref}_p", f"|F|=8 fixed-cadence vs {ref}, raw p", fmt_p(p), "two-sided paired t-test")
        add(f"lm_fcr8_vs_{ref}_pholm", f"|F|=8 fixed-cadence vs {ref}, Holm p", fmt_p(pa),
            "Holm over the three fixed-cadence contrasts at |F|=8")
    # TOST fixed-cadence vs round-robin at |F|=8, final endpoint
    s8 = sorted(set(fin(FCR, 8)) & set(fin(RR, 8)))
    d8 = np.array([fin(FCR, 8)[s] - fin(RR, 8)[s] for s in s8])
    margin = 0.01 * float(np.mean(list(fin(RR, 8).values())))
    p_tost, lo90, hi90, p_lo, p_hi = tost_paired(d8, margin)
    add("lm_tost_fcr8_rr_margin", "TOST margin, |F|=8 fixed-cadence vs round-robin (ppl)", fmt(margin, 3),
        "1% of the round_robin |F|=8 mean final test ppl")
    add("lm_tost_fcr8_rr_p", "TOST p, |F|=8 fixed-cadence vs round-robin, final", fmt_p(p_tost),
        f"max of two one-sided paired t-tests, df=n-1 (n={len(s8)}); p(lower)={p_lo:.4g}, p(upper)={p_hi:.4g}")
    add("lm_tost_fcr8_rr_ci90_lo", "90% CI of fixed-cadence minus round-robin at |F|=8, lower (ppl)", fmt(lo90, 3),
        "paired t interval, df=n-1")
    add("lm_tost_fcr8_rr_ci90_hi", "90% CI of fixed-cadence minus round-robin at |F|=8, upper (ppl)", fmt(hi90, 3), "same")
    md, pos, p = contrast(fin("learned_top1_aux0", 8), fin("learned_top1", 8))
    add("lm_aux0_vs_learned8_p", "|F|=8 learned aux0 vs learned, p", fmt(p, 4), f"two-sided paired t-test (diff {md:.2f})")
    D = defaultdict(dict)
    for r in lm:
        if r["prefix"] == lm_wide:
            D[r["method"]][r["seed"]] = r
    if RR not in D or "learned_top1" not in D:
        sys.exit(f"--lm-wide {lm_wide}: round_robin or learned_top1 runs missing")
    md, pos, p = contrast({s: r["final"] for s, r in D["learned_top1"].items()},
                          {s: r["final"] for s, r in D[RR].items()})
    add("d256_learned_minus_rr", "d=256 learned minus round-robin (ppl)", fmt(md, 2), f"LM grid {lm_wide}_*; mean paired difference")
    add("d256_learned_pos", "d=256 seeds with learned > round-robin", pos, "count")
    add("d256_learned_p", "d=256 learned vs round-robin p", fmt(p, 4), "two-sided paired t-test")
    for lab, key in (("final", "final"), ("bestval", "at_best_val")):
        refs = ("learned_top1", FCR, "random")
        cs = [contrast({s: r[key] for s, r in D[m].items()}, {s: r[key] for s, r in D[RR].items()}) for m in refs]
        adj = holm([c[2] for c in cs])
        for m, (md, pos, p), pa in zip(refs, cs, adj):
            what = "final test ppl" if lab == "final" else "test ppl at the best-validation epoch"
            add(f"d256_{m}_vs_rr_{lab}_diff", f"d=256 {m} minus round-robin, {lab} (ppl)", fmt(md, 2),
                f"mean paired difference of {what}")
            add(f"d256_{m}_vs_rr_{lab}_pos", f"d=256 seeds with {m} > round-robin, {lab}", f"{pos}/{len(set(D[m]) & set(D[RR]))}",
                "count / paired seeds")
            add(f"d256_{m}_vs_rr_{lab}_p", f"d=256 {m} vs round-robin, raw p, {lab}", fmt_p(p), "two-sided paired t-test")
            add(f"d256_{m}_vs_rr_{lab}_pholm", f"d=256 {m} vs round-robin, Holm p, {lab}", fmt_p(pa),
                f"Holm over the three d=256 contrasts vs round_robin ({lab})")
    means = [np.mean([r["final"] for r in D[m].values()]) for m in D]
    add("d256_arm_min", "d=256 arm means, min (ppl)", fmt(min(means), 2), f"over {len(D)} arms")
    add("d256_arm_max", "d=256 arm means, max (ppl)", fmt(max(means), 2), "same")
    mc = np.mean([r["curve"] for r in D["learned_top1"].values()], axis=0)
    i = int(np.argmin(mc))
    add("d256_learned_curve_min", "d=256 learned seed-mean curve minimum (ppl)", fmt(mc[i], 2), "mean over seeds per epoch")
    add("d256_learned_curve_postmax", "d=256 learned seed-mean curve maximum after its minimum (ppl)", fmt(mc[i:].max(), 2), "same")
    bv = lambda m, F: {s: r["at_best_val"] for s, r in T[(m, F)].items()}
    for lab, getter in (("final", fin), ("bestval", bv)):
        md, pos, p = contrast(getter(FCR, 4), getter(RR, 4))
        add(f"lm_fcr4_{lab}_diff", f"|F|=4 fixed-cadence minus round-robin, {lab} (ppl)", fmt(md, 2),
            "mean paired difference" + (" of test ppl at the best-validation epoch" if lab == "bestval" else " of final test ppl"))
        add(f"lm_fcr4_{lab}_pos", f"|F|=4 seeds with fixed-cadence > round-robin, {lab}", pos, "count")
        add(f"lm_fcr4_{lab}_p", f"|F|=4 fixed-cadence vs round-robin p, {lab}", fmt_p(p), "two-sided paired t-test")
    for m in sorted(m for m, F in T if F == 4):
        add(f"lm_gapcv_F4_{m}", f"gap-CV at |F|=4, {m}", fmt(np.mean([gap_cv(routes(r)) for r in T[(m, 4)].values()]), 3),
            "as in lm_cv_pearson")
    for m in (RR, FCR):
        pooled = Counter()
        for r in T[(m, 4)].values():
            pooled.update(gap_hist(routes(r)))
        add(f"lm_gaphist_F4_{m}", f"pooled inter-activation gap counts at |F|=4, {m}",
            json.dumps({int(k): int(v) for k, v in sorted(pooled.items())}),
            f"{src}; gap (steps) -> count over all seeds, layers and non-null experts; total {sum(pooled.values())}")

    # ── 7 token level ─────────────────────────────────────────────────────────────
    e5 = [r for r in ok if r["prefix"] == "e5"]
    c = e5[0]["cfg"]
    b, S, dm, fs, W = c["batch_size"], (32 // PATCH["e5"]) ** 2, c["d_model"], c["d_ff_shared"], c["d_ff_expert"]
    bound = chat(b, S, dm, fs) + sum(2 * TAU * (2 * dm * w + w + dm) for w in W)
    add("tok_bound_KiB", "capacity-router co-active lower bound (KiB)", fmt(bound / 1024, 2),
        "C(shared) + sum over experts of 2P_e (parameters + gradients), token-level backbone")
    budgets = sorted({r["cfg"]["budget_bytes"] for r in e5})
    for i, bg in enumerate(budgets):
        add(f"tok_budget_{i + 1}_KiB", f"token-level budget {i + 1} (KiB)", fmt(bg / 1024, 2), "config budget_bytes / 1024")
    capk = defaultdict(list)
    for r in e5:
        if r["method"] == "token_capacity":
            capk[r["cfg"]["budget_bytes"]].append(max(r["peaks"]))
    allp = [p for v in capk.values() for p in v]
    add("tok_cap_peak_min_KiB", "capacity-router run-level peak, min (KiB)", fmt(min(allp) / 1024, 2), "max over epochs per run")
    add("tok_cap_peak_max_KiB", "capacity-router run-level peak, max (KiB)", fmt(max(allp) / 1024, 2), "same")
    for i, bg in enumerate(budgets[:2]):
        add(f"tok_cap_peak_x_budget_{i + 1}", f"capacity peak as multiple of budget {i + 1}", fmt(max(capk[bg]) / bg, 3),
            "max run-level peak / budget")
    kn = [r for r in e5 if r["method"] == "token_knapsack"]
    add("tok_knap_violations", "knapsack runs with peak > budget", f"{sum(max(r['peaks']) > r['cfg']['budget_bytes'] for r in kn)}/{len(kn)}", "count")
    top = budgets[-1]
    for ds in ("c10", "c100"):
        A = {r["seed"]: r["final"] for r in e5 if r["dataset"] == ds and r["method"] == "token_knapsack" and r["cfg"]["budget_bytes"] == top}
        B = {r["seed"]: r["final"] for r in e5 if r["dataset"] == ds and r["method"] == "token_capacity" and r["cfg"]["budget_bytes"] == top}
        md, pos, p = contrast(A, B)
        add(f"tok_knap_vs_cap_{ds}_p", f"knapsack vs capacity at the compliant budget, {ds}, p", fmt(p, 4),
            f"two-sided paired t-test, n={len(set(A) & set(B))}, failed run included")
        add(f"tok_knap_vs_cap_{ds}_diff", f"knapsack minus capacity, {ds} (pp)", fmt(md * 100, 3), "mean paired difference")
    lo = stats.beta.ppf(0.025, 1, 30)
    hi = stats.beta.ppf(0.975, 2, 29)
    add("tok_fail_ci_lo", "Clopper-Pearson 95% CI for 1/30, lower (%)", fmt(lo * 100, 3), "exact binomial")
    add("tok_fail_ci_hi", "Clopper-Pearson 95% CI for 1/30, upper (%)", fmt(hi * 100, 3), "exact binomial")
    cf = defaultdict(list)
    for r in ok:
        if r["prefix"] == "e5tune":
            cf[r["cell"]].append(r["best_val_acc"])
    for k in sorted(cf):
        add(f"tok_tune_{k}", f"capacity-factor tuning, {k}: mean best validation accuracy", fmt(np.mean(cf[k]), 4),
            f"mean over datasets (n={len(cf[k])}) of best_val_acc, one seed")

    # ── 8 memory ─────────────────────────────────────────────────────────────────
    e3 = json.load(open(E3_JSON))["blocks"]
    add("mem_saved_le_cplus", "configs with saved-tensor bytes + 2P <= C+", f"{sum(x['live_model_bytes'] <= x['chat_plus'] for x in e3)}/{len(e3)}",
        "live_model_bytes (params + grads + saved tensors) vs chat_plus")
    slack = [1 - x["live_model_bytes"] / x["chat_plus"] for x in e3]
    add("mem_slack_min", "saved-tensor slack below C+, min (%)", fmt(min(slack) * 100, 1), "1 - (saved + 2P)/C+")
    add("mem_slack_max", "saved-tensor slack below C+, max (%)", fmt(max(slack) * 100, 1), "same")
    med = lambda x: float(np.median(x["measured_alloc_peak_all"]))
    add("mem_alloc_gt_cplus", "configs with allocated peak > C+", f"{sum(med(x) > x['chat_plus'] for x in e3)}/{len(e3)}",
        "median over repeats of the framework allocated peak")
    fam = defaultdict(list)
    for x in e3:
        fam[(x["family"], x["inplace"])].append(med(x) / x["chat_plus"] - 1)
    for (f, ip), v in sorted(fam.items()):
        tag = f"{f}_inplace{int(ip)}"
        add(f"mem_excess_{tag}_min", f"allocated-peak excess over C+, {tag}, min (%)", fmt(min(v) * 100, 1), "median alloc / C+ - 1")
        add(f"mem_excess_{tag}_max", f"allocated-peak excess over C+, {tag}, max (%)", fmt(max(v) * 100, 1), "same")
    ratio = []
    for x in e3:
        if x["inplace"]:
            base = next(y for y in e3 if y["family"] == x["family"] and y["f"] == x["f"] and not y["inplace"])
            ratio.append(med(x) / med(base))
    add("mem_inplace_ratio_max", "in-place / non-in-place allocated peak, max", fmt(max(ratio), 3), "median alloc, same (family, width)")
    add("mem_inplace_ratio_gt1", "widths where in-place increases the allocated peak", f"{sum(r > 1 for r in ratio)}/{len(ratio)}", "ratio > 1.0")
    for f in sorted({x["family"] for x in e3}):
        xs = [x for x in e3 if x["family"] == f and not x["inplace"]]
        rr = [delta_ln(x["b"], x["S"], x["d"]) / chat(x["b"], x["S"], x["d"], x["f"]) for x in xs]
        add(f"mem_dln_{f}_min", f"Delta_LN / C over widths, {f}, min (%)", fmt(min(rr) * 100, 1), "Delta_LN = 2 tau b S d + 4 tau b S")
        add(f"mem_dln_{f}_max", f"Delta_LN / C over widths, {f}, max (%)", fmt(max(rr) * 100, 1), "same")
    ft = max(x["measured_alloc_peak_all"][0] - med(x) for x in e3)
    add("mem_first_touch_MiB", "first-repeat excess over the median, max over configs (MiB)", fmt(ft / 2**20, 2),
        "one-time framework initialization captured in the first repeat")

    # ── 9 parameter counts ─────────────────────────────────────────────────────────
    lin = lambda i, o: i * o + o

    def vis_params(r, L=4):
        c = r["cfg"]
        d, p = c["d_model"], PATCH[r["prefix"]]
        widths = c["d_ff_expert"] if isinstance(c["d_ff_expert"], list) else [c["d_ff_expert"]] * c["num_experts"]
        C = 100 if c["dataset"] == "cifar100" else 10
        layer = 4 * d + lin(d, c["d_ff_shared"]) + lin(c["d_ff_shared"], d) + sum(lin(d, w) + lin(w, d) for w in widths)
        return 3 * p * p * d + d + (32 // p) ** 2 * d + L * layer + lin(d, C)

    def lm_params(c):
        d, fs, V = c["d_model"], c["d_ff_shared"], c["vocab_size"]
        layer = 6 * d + lin(d, 3 * d) + lin(d, d) + lin(d, fs) + lin(fs, d) + sum(lin(d, w) + lin(w, d) for w in c["widths"])
        emb_head = V * d + c["bptt"] * d + lin(d, V)
        return emb_head + c["num_layers"] * layer + 2 * d, emb_head

    pick = lambda pred: next(r for r in ok if pred(r))
    add("params_s1", "parameters, S1 backbone (CIFAR-100)", vis_params(pick(lambda r: r["prefix"] == "s1")),
        "patch conv + position table + 4 x (2 LayerNorm, shared FFN, 8 experts) + head; router excluded")
    for ds in ("c10", "c100"):
        add(f"params_s2_{ds}", f"parameters, S2 backbone ({ds})", vis_params(pick(lambda r: r["prefix"] == "e1" and r["dataset"] == ds)), "same")
    add("params_e4", "parameters, device-scale backbone", vis_params(pick(lambda r: r["prefix"] == "e4")), "same")
    p128, eh128 = lm_params(pick(lambda r: r["prefix"] == lm_main)["cfg"])
    p256, _ = lm_params(pick(lambda r: r["prefix"] == lm_wide)["cfg"])
    add("params_lm_d128", "parameters, LM d=128", p128, f"{lm_main} config; embedding + positions + 4 blocks + final LayerNorm + untied head")
    add("params_lm_d128_embhead", "parameters, LM d=128, embedding + positions + head", eh128, "same")
    add("params_lm_d256", "parameters, LM d=256", p256, f"{lm_wide} config; same")

    # ── 10 determinism ─────────────────────────────────────────────────────────────
    diff = [p for p in tie if p["dis"] == 0 and p["dacc"] != 0]
    add("det_rrdb_vision_differ", "identical-trace round_robin/gatedmoe_base vision pairs with different final acc",
        f"{len(diff)}/{len(tie)}", "exact inequality")
    add("det_rrdb_vision_maxabs", "max |final-acc difference| among them", fmt(max((abs(p["dacc"]) for p in diff), default=0), 4), "accuracy units")
    add("det_rrdb_vision_seeds", "seeds of those pairs", ";".join(sorted({str(p["seed"]) for p in diff})), "")
    lm_same = lm_tot = 0
    for F in (0, 4, 8):
        A, B = T[(RR, F)], T[(DB, F)]
        s = sorted(set(A) & set(B))
        if not s:
            sys.exit(f"--lm-main {lm_main}: no round_robin/gatedmoe_base pairs at |F|={F}")
        lm_tot += len(s)
        lm_same += sum(np.array_equal(routes(A[i]), routes(B[i])) for i in s)
        dd = [abs(A[i]["final"] - B[i]["final"]) for i in s]
        add(f"det_rrdb_lm_F{F}_differ", f"LM |F|={F}: seeds with different final ppl, round_robin vs gatedmoe_base",
            f"{sum(x != 0 for x in dd)}/{len(s)}", f"{lm_main}; exact inequality")
        add(f"det_rrdb_lm_F{F}_maxabs", f"LM |F|={F}: max |final-ppl difference|", fmt(max(dd), 3), "ppl")
    add("rrdb_lm_identical", "round_robin vs gatedmoe_base LM trace pairs identical", f"{lm_same}/{lm_tot}",
        f"{lm_main}; exact equality of routes_train.npy")
    twin = defaultdict(list)
    for r in e5:
        if r["method"] == "token_capacity":
            twin[(r["dataset"], r["seed"])].append(r["final"])
    add("det_capacity_twin_maxabs", "capacity control: max |final-acc difference| across budgets, identical configs",
        fmt(max(max(v) - min(v) for v in twin.values()), 4), "the capacity router has no memory gate, so runs differing only in budget are identical configs")

    # ── 11 rerun comparison (superseded vs corrected) ─────────────────────────────
    def lm_key(r):
        return (r["method"], int(r["cfg"]["d_model"]), r["budget_kb"], r["seed"])

    pairs = []  # (old, new)
    for old_p, new_p in zip(SUPERSEDED, CORRECTED):
        old = {lm_key(r): r for r in lm if r["prefix"] == old_p}
        new = {lm_key(r): r for r in lm if r["prefix"] == new_p}
        if not old or set(old) != set(new):
            sys.exit(f"rerun comparison: {old_p} ({len(old)}) and {new_p} ({len(new)}) do not pair one-to-one")
        pairs += [(old[k], new[k]) for k in sorted(old)]
    add("rerun_pairs", "superseded/corrected run pairs", len(pairs),
        f"{'+'.join(f'{a}<->{b}' for a, b in zip(SUPERSEDED, CORRECTED))}, keyed by (method, d_model, budget, seed)")

    def f_of(r):
        return LM_F.get(r["budget_kb"])  # None for d=256 (all-feasible budget)

    unaff_methods = {RR, "random", "learned_top1", "learned_top1_aux0"}
    unaff = [(o, n) for o, n in pairs if o["method"] in unaff_methods
             or (o["method"] in (DB, DG) and o["cfg"]["d_model"] == 128 and f_of(o) == 0)]
    ident = [(o, n) for o, n in unaff if o["final"] == n["final"] and o["at_best_val"] == n["at_best_val"]]
    add("rerun_unaffected_identical", "unaffected runs identical in the rerun", f"{len(ident)}/{len(unaff)}",
        "round_robin, random, learned_top1, learned_top1_aux0 (all budgets and d) and gatedmoe_base/gatedmoe_g at |F|=0; "
        "identical = final_test_ppl and test_ppl_at_best_val exactly equal")
    ident_names = {n["name"] for o, n in ident}
    nonid = [(o, n) for o, n in unaff if n["name"] not in ident_names]
    add("rerun_unaffected_nonidentical", "unaffected runs that differ in the rerun",
        ";".join(f"{n['name']}:{abs(n['final'] - o['final']):.3f}" for o, n in nonid),
        "corrected run name : |final_test_ppl difference|")
    affected = [(FCR, 128, 4), (FCR, 128, 8), (FCR, 256, None), ("cycle_shuffle", 128, 8),
                (DB, 128, 4), (DB, 128, 8), (DG, 128, 4), (DG, 128, 8)]
    n_aff = 0
    for m, dm_, F in affected:
        sel = [(o, n) for o, n in pairs if o["method"] == m and o["cfg"]["d_model"] == dm_ and f_of(o) == F]
        if not sel:
            sys.exit(f"rerun comparison: no pairs for {m} d={dm_} |F|={F}")
        n_aff += len(sel)
        tag = f"{m}_{'d256' if dm_ == 256 else f'F{F}'}"
        dfin = np.array([n["final"] - o["final"] for o, n in sel])
        dbv = np.array([n["at_best_val"] - o["at_best_val"] for o, n in sel])
        add(f"rerun_{tag}_final_diff", f"rerun {tag}: mean(corrected - superseded) final ppl", fmt(dfin.mean(), 3),
            f"paired by seed, n={len(sel)}")
        add(f"rerun_{tag}_nlower", f"rerun {tag}: seeds with corrected < superseded, final", f"{int((dfin < 0).sum())}/{len(sel)}", "count")
        add(f"rerun_{tag}_p", f"rerun {tag}: p, final", fmt_p(paired_p([n["final"] for o, n in sel], [o["final"] for o, n in sel])),
            "two-sided one-sample t-test on the paired differences")
        add(f"rerun_{tag}_bestval_diff", f"rerun {tag}: mean(corrected - superseded) test ppl at best val", fmt(dbv.mean(), 3),
            "paired by seed")
    add("rerun_classified", "rerun pairs classified (unaffected + affected) / all", f"{len(unaff) + n_aff}/{len(pairs)}",
        "every pair belongs to exactly one class")
    Tsup = defaultdict(dict)
    for r in lm:
        if r["prefix"] == SUPERSEDED[0]:
            Tsup[(r["method"], LM_F[r["budget_kb"]])][r["seed"]] = r
    for F in (0, 4, 8):
        A, B = Tsup[(RR, F)], Tsup[(DB, F)]
        s = sorted(set(A) & set(B))
        add(f"rerun_superseded_rrdb_F{F}_maxabs", f"superseded {SUPERSEDED[0]} |F|={F}: max |final ppl gatedmoe_base - round_robin|",
            fmt(max(abs(A[i]["final"] - B[i]["final"]) for i in s), 3),
            f"n={len(s)} seeds; routes_train.npy identical in {sum(np.array_equal(routes(A[i]), routes(B[i])) for i in s)}/{len(s)}; "
            + ("no routing at |F|=0, so the evaluation defect cannot act here: this is run-to-run GPU "
               "nondeterminism (one superseded round_robin run; the corrected rerun matches its twins)"
               if F == 0 else "evaluation-defect effect (PROTOCOL.md Amendment 4)"))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "quantity", "value", "definition"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"{len(ROWS)} rows -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")


if __name__ == "__main__":
    main()
