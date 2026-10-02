"""Language-model (LM) analysis tables, parameterized by run prefix.

Reads per-run files only (result/logs/<run>/results.json, routes_train.npy):
  main backbone d=128:  <prefix>_<method>_B<budget>k_s<seed>
                        budgets 17991k / 24457k / 42487k -> |F| = 0 / 4 / 8
  wide backbone d=256:  <wide>_<method>_B63462k_s<seed>, |F| = 8
                        wide = e6lmd256 for prefix e6lm2, e6lm3d256 for prefix e6lm3
and writes into --out-dir (refuses to overwrite any existing output):
  lm_table.csv     per (backbone, |F|, method): n, mean and sample SD (ddof=1) of the final-epoch
                   test ppl (primary) and the test ppl at the best-validation epoch (secondary),
                   LaTeX cells rounded half-up to one decimal, mean best-validation epoch
                   (1-indexed), mean final-epoch train_ppl and train_obj_ppl (when logged).
  lm_tests.csv     paired-by-seed tests (two-sided paired t); Holm families; TOST.
  lm_dynamics.csv  seed-mean test-ppl curves and their extrema, utilization entropy, gap-CV,
                   pooled gap-value counts, arm-level correlations of final ppl with gap-CV.
  lm_curves.csv    long format, one row per (run, epoch).
  lm_summary.csv   one row per run.
  debt_vs_rr.csv   per |F| and seed: debt-base routes equal to round-robin routes, ppl difference.
All epoch numbers in the outputs are 1-indexed (results.json stores 0-indexed epochs).

Usage (from the repository root):
  python3 scripts/analyze-lm-v11.py --prefix e6lm3 --out-dir result/tables/lm_v11
Options: --logs-dir (default result/logs), --allow-incomplete (analyse finished runs only).
Dependencies: numpy, scipy.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
WIDE = {"e6lm2": "e6lmd256", "e6lm3": "e6lm3d256"}
PROTOCOL = {"e6lm3": "s0-lm-a4", "e6lm3d256": "s0-lm-a4"}  # required config.protocol, if listed
F_MAIN = {17991: 0, 24457: 4, 42487: 8}
F_WIDE = {63462: 8}
RR = "round_robin"
DB = "gatedmoe_base"  # debt-based arm whose schedule duplicates round-robin
FCR = "fixed_cadence_random"
LEARNED = "learned_top1"
AUX0 = "learned_top1_aux0"
FAMILY12 = ("gatedmoe_base", "gatedmoe_g", "random", "learned_top1")
ENDPOINTS = ("final", "bestval")  # final-epoch test ppl; test ppl at the best-validation epoch
OUTS = ("lm_table.csv", "lm_tests.csv", "lm_dynamics.csv", "lm_curves.csv", "lm_summary.csv",
        "debt_vs_rr.csv")


# ── helpers ───────────────────────────────────────────────────────────────────
def r1(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def holm(p):
    p = np.asarray(p, float)
    adj, cur = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p, kind="mergesort")):
        cur = max(cur, (len(p) - k) * p[i])
        adj[i] = min(1.0, cur)
    return adj


def paired(A, B):
    """A, B: seed -> value. Paired over common seeds; exactly identical vectors -> p = 1."""
    s = sorted(set(A) & set(B))
    a = np.array([A[i] for i in s], float)
    b = np.array([B[i] for i in s], float)
    d = a - b
    p = 1.0 if np.all(d == 0) else float(stats.ttest_rel(a, b).pvalue)
    return {"n": len(s), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "pos_seeds": int((d > 0).sum()), "p": p, "_d": d}


def tost(d, eps):
    """Two one-sided paired t tests against +-eps; p = max; 90% CI of the mean difference."""
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        p = 0.0 if abs(m) < eps else 1.0
        return p, m, m
    p_lo = float(stats.t.sf((m + eps) / se, n - 1))   # H0: mean <= -eps
    p_hi = float(stats.t.cdf((m - eps) / se, n - 1))  # H0: mean >= +eps
    h = float(stats.t.ppf(0.95, n - 1)) * se
    return max(p_lo, p_hi), m - h, m + h


def entropy_from_routes(rt):
    out = []
    for l in range(rt.shape[1]):
        c = np.bincount(rt[:, l][rt[:, l] >= 0].astype(int), minlength=8).astype(float)
        if c.sum() == 0:
            out.append(0.0)
            continue
        q = c[c > 0] / c.sum()
        out.append(float(-(q * np.log(q)).sum()))
    return out


def gap_cv(rt):
    """Mean over (layer, expert with >2 activations) of std(gaps, ddof=0)/mean(gaps)."""
    cvs = []
    for l in range(rt.shape[1]):
        for e in range(8):
            idx = np.where(rt[:, l] == e)[0]
            if len(idx) > 2:
                g = np.diff(idx)
                cvs.append(g.std() / g.mean())
    return float(np.mean(cvs))


def gap_counts(rt):
    """Counts of inter-activation gaps (steps), all layers and experts."""
    vals = Counter()
    for l in range(rt.shape[1]):
        col = rt[:, l]
        for e in np.unique(col[col >= 0]):
            vals.update(np.diff(np.where(col == e)[0]).tolist())
    return vals


# ── loading ───────────────────────────────────────────────────────────────────
def load(logs, prefix, allow_incomplete):
    runs, missing = {}, []
    for grid, backbone, fmap in ((prefix, 128, F_MAIN), (WIDE[prefix], 256, F_WIDE)):
        pat = re.compile(rf"^{grid}_(\w+?)_B(\d+)k_s(\d+)$")
        for d in sorted(logs.iterdir()):
            m = pat.match(d.name)
            if not m or not d.is_dir():
                continue
            method, bkb, seed = m.group(1), int(m.group(2)), int(m.group(3))
            if bkb not in fmap:
                sys.exit(f"unexpected budget {bkb}k in {d.name}")
            f = d / "results.json"
            if not f.exists():
                missing.append(d.name)
                continue
            j = json.load(open(f))
            cfg, eps = j["config"], j["epochs"]
            F = fmap[bkb]
            if len(eps) != cfg["epochs"] or cfg["d_model"] != backbone:
                sys.exit(f"{d.name}: {len(eps)} epochs (config {cfg['epochs']}), d_model {cfg['d_model']}")
            if grid in PROTOCOL and cfg.get("protocol") != PROTOCOL[grid]:
                sys.exit(f"{d.name}: protocol {cfg.get('protocol')!r} != {PROTOCOL[grid]!r}")
            fh = np.array([e["feasible_hist"] for e in eps]).sum(axis=0)
            if [int(i) for i in np.nonzero(fh)[0]] != [F]:
                sys.exit(f"{d.name}: realized |F| {np.nonzero(fh)[0].tolist()} != {F}")
            runs[(backbone, F, method, seed)] = {
                "name": d.name, "dir": d, "grid": grid, "budget_kb": bkb,
                "final": j["final_test_ppl"], "bestval": j["test_ppl_at_best_val"],
                "bve": j["best_val_epoch"] + 1, "epochs": eps}
    if missing:
        print(f"runs without results.json: {len(missing)}: {', '.join(missing)}")
        if not allow_incomplete:
            sys.exit("incomplete grid (use --allow-incomplete to analyse finished runs only)")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, choices=sorted(WIDE))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--logs-dir", default=str(ROOT / "result" / "logs"))
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    for n in OUTS:
        if (out_dir / n).exists():
            sys.exit(f"refusing to overwrite {out_dir / n}")
    runs = load(Path(a.logs_dir), a.prefix, a.allow_incomplete)
    if not runs:
        sys.exit("no finished runs")
    route_cache = {}

    def routes(r):
        if r["name"] not in route_cache:
            route_cache[r["name"]] = np.load(r["dir"] / "routes_train.npy")
        return route_cache[r["name"]]

    cells = defaultdict(dict)  # (backbone, F, method) -> seed -> run
    for (bb, F, m, s), r in sorted(runs.items(), key=lambda kv: kv[1]["name"]):
        cells[(bb, F, m)][s] = r
    print(f"prefix {a.prefix} (wide {WIDE[a.prefix]}): {len(runs)} finished runs, {len(cells)} cells")

    def ep_vals(bb, F, m, key):
        return {s: r[key] for s, r in cells[(bb, F, m)].items()}

    # ── lm_table ──────────────────────────────────────────────────────────────
    table = []
    for (bb, F, m) in sorted(cells):
        rs = list(cells[(bb, F, m)].values())
        fin = np.array([r["final"] for r in rs])
        bv = np.array([r["bestval"] for r in rs])
        sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else float("nan")
        obj = [r["epochs"][-1].get("train_obj_ppl") for r in rs]
        row = {"backbone": bb, "F": F, "method": m, "budget_kb": rs[0]["budget_kb"], "n": len(rs),
               "final_mean": float(fin.mean()), "final_sd": sd(fin),
               "bestval_mean": float(bv.mean()), "bestval_sd": sd(bv),
               "final_cell": f"{r1(fin.mean())}$\\pm${r1(sd(fin))}",
               "bestval_cell": f"{r1(bv.mean())}$\\pm${r1(sd(bv))}",
               "best_val_epoch_mean_1idx": float(np.mean([r["bve"] for r in rs])),
               "final_train_ppl_mean": float(np.mean([r["epochs"][-1]["train_ppl"] for r in rs])),
               "final_train_obj_ppl_mean": float(np.mean(obj)) if all(o is not None for o in obj) else ""}
        table.append(row)
        print(f"d={bb} |F|={F} {m:22s} n={len(rs):2d} final {row['final_cell']:20s} bestval {row['bestval_cell']:18s}"
              f" bve {row['best_val_epoch_mean_1idx']:.1f} train {row['final_train_ppl_mean']:.2f}"
              + (f" obj {row['final_train_obj_ppl_mean']:.2f}" if row["final_train_obj_ppl_mean"] != "" else ""))

    # ── lm_tests ──────────────────────────────────────────────────────────────
    tests = []

    def add_test(test, ep, bb, F, ma, mb, family=""):
        res = paired(ep_vals(bb, F, ma, ep), ep_vals(bb, F, mb, ep))
        row = {"test": test, "endpoint": ep, "backbone": bb, "F": F, "method_a": ma, "method_b": mb,
               "n": res["n"], "mean_diff": res["mean_diff"], "sd_diff": res["sd_diff"],
               "pos_seeds": res["pos_seeds"], "p": res["p"], "holm_family": family, "holm_m": "",
               "p_holm": "", "margin": "", "p_tost": "", "ci90_lo": "", "ci90_hi": ""}
        row["_d"] = res["_d"]
        tests.append(row)
        return row

    def holm_over(rows, family, expect_m):
        if len(rows) != expect_m:
            sys.exit(f"Holm family {family}: {len(rows)} tests, expected {expect_m}")
        for r, pa in zip(rows, holm([r["p"] for r in rows])):
            r.update(holm_family=family, holm_m=len(rows), p_holm=float(pa))

    for ep in ENDPOINTS:
        fam = []
        for (bb, F, m) in sorted(cells):
            if bb == 128 and m != RR and (bb, F, RR) in cells:
                row = add_test("vs_rr", ep, bb, F, m, RR)
                if m in FAMILY12:
                    fam.append(row)
        holm_over(fam, "table12", 12)
        holm_over([add_test("fcr8_intervention", ep, 128, 8, FCR, ref) for ref in (RR, "random", LEARNED)],
                  "fcr8_three", 3)
        t = add_test("fcr8_tost", ep, 128, 8, FCR, RR)
        eps = 0.01 * float(np.mean(list(ep_vals(128, 8, RR, ep).values())))
        p_t, lo, hi = tost(t["_d"], eps)
        t.update(margin=eps, p_tost=p_t, ci90_lo=lo, ci90_hi=hi)
        add_test("aux0_vs_learned8", ep, 128, 8, AUX0, LEARNED)
        holm_over([add_test("wide_vs_rr", ep, 256, 8, m, RR) for m in (FCR, "random", LEARNED)],
                  "wide_three", 3)
        add_test("fcr4_vs_rr", ep, 128, 4, FCR, RR)
    for t in tests:
        print(f"{t['test']:18s} {t['endpoint']:7s} d={t['backbone']} |F|={t['F']} {t['method_a']:22s}- {t['method_b']:14s}"
              f" n={t['n']:2d} diff {t['mean_diff']:+8.2f} pos {t['pos_seeds']:2d} p={t['p']:.4g}"
              + (f" holm[{t['holm_family']}]={t['p_holm']:.4g}" if t["p_holm"] != "" else "")
              + (f" margin={t['margin']:.3f} p_tost={t['p_tost']:.4g} ci90=[{t['ci90_lo']:.2f},{t['ci90_hi']:.2f}]"
                 if t["margin"] != "" else ""))

    # ── lm_dynamics ───────────────────────────────────────────────────────────
    dyn = []

    def D(quantity, bb, F, method, value, seed="", epoch="", key="", note=""):
        dyn.append({"quantity": quantity, "backbone": bb, "F": F, "method": method, "seed": seed,
                    "epoch": epoch, "key": key, "value": value, "note": note})

    for bb in (128, 256):
        for m in sorted(m for (b, F, m) in cells if b == bb and F == 8):
            rs = list(cells[(bb, 8, m)].values())
            mc = np.mean([[e["test_ppl"] for e in r["epochs"]] for r in rs], axis=0)
            for i, v in enumerate(mc):
                D("curve_mean_test_ppl", bb, 8, m, float(v), epoch=i + 1, note=f"mean over {len(rs)} seeds")
            i = int(np.argmin(mc))
            D("curve_min", bb, 8, m, float(mc[i]))
            D("curve_argmin_epoch", bb, 8, m, i + 1)
            D("curve_postmin_max", bb, 8, m, float(mc[i:].max()), note="max of the seed-mean curve from its minimum on")
            D("curve_final", bb, 8, m, float(mc[-1]))
            D("final_train_ppl_mean", bb, 8, m, float(np.mean([r["epochs"][-1]["train_ppl"] for r in rs])))
            ents, src = [], set()
            for r in rs:
                ds = r["epochs"][-1].get("debt_stats")
                if ds and all("utilization_entropy" in x for x in ds):
                    ents += [float(x["utilization_entropy"]) for x in ds]
                    src.add("debt_stats")
                else:
                    ents += entropy_from_routes(routes(r))
                    src.add("routes_train")
            D("entropy_min", bb, 8, m, min(ents), note="final epoch, over seeds and layers; source " + "+".join(sorted(src)))
            D("entropy_max", bb, 8, m, max(ents), note="final epoch, over seeds and layers; source " + "+".join(sorted(src)))
    arm_cv = {}
    for F in (8, 4):
        for m in sorted(m for (b, f, m) in cells if b == 128 and f == F):
            vals = []
            for s, r in sorted(cells[(128, F, m)].items()):
                v = gap_cv(routes(r))
                vals.append(v)
                D("gapcv_run", 128, F, m, v, seed=s)
            arm_cv[(F, m)] = float(np.mean(vals))
            D("gapcv_arm", 128, F, m, arm_cv[(F, m)], note=f"mean over {len(vals)} seeds")
    for m in (RR, FCR):
        pooled = Counter()
        for r in cells[(128, 4, m)].values():
            pooled.update(gap_counts(routes(r)))
        for g, c in sorted(pooled.items()):
            D("gap_count_pooled", 128, 4, m, c, key=g, note="pooled over seeds, layers and experts")
    arms8 = sorted(m for (b, f, m) in cells if b == 128 and f == 8)
    for label, arms in (("all", arms8), ("excl_" + DB, [m for m in arms8 if m != DB])):
        x = [arm_cv[(8, m)] for m in arms]
        y = [float(np.mean([r["final"] for r in cells[(128, 8, m)].values()])) for m in arms]
        note = f"{len(arms)} arms: " + ";".join(arms)
        D("corr_pearson_ppl_gapcv", 128, 8, label, float(stats.pearsonr(x, y)[0]), note=note)
        D("corr_spearman_ppl_gapcv", 128, 8, label, float(stats.spearmanr(x, y)[0]), note=note)
    for r in dyn:
        if not r["quantity"].startswith(("curve_mean", "gapcv_run")):
            print(f"{r['quantity']:24s} d={r['backbone']} |F|={r['F']} {r['method']:22s} {r['key']!s:4s} {r['value']}")

    # ── lm_curves, lm_summary ─────────────────────────────────────────────────
    curves, summary = [], []
    for (bb, F, m, s), r in sorted(runs.items(), key=lambda kv: kv[1]["name"]):
        summary.append({"method": m, "backbone": bb, "F": F, "seed": s, "final_test_ppl": r["final"],
                        "test_ppl_at_best_val": r["bestval"], "best_val_epoch": r["bve"]})
        for e in r["epochs"]:
            curves.append({"method": m, "backbone": bb, "F": F, "seed": s, "epoch": e["epoch"] + 1,
                           "test_ppl": e["test_ppl"], "val_ppl": e["val_ppl"], "train_ppl": e["train_ppl"]})

    # ── debt_vs_rr ────────────────────────────────────────────────────────────
    dvr = []
    for F in (0, 4, 8):
        A, B = cells.get((128, F, DB), {}), cells.get((128, F, RR), {})
        for s in sorted(set(A) & set(B)):
            eq = bool(np.array_equal(routes(A[s]), routes(B[s])))
            dvr.append({"F": F, "seed": s, "routes_equal": eq,
                        "final_diff": A[s]["final"] - B[s]["final"],
                        "bestval_diff": A[s]["bestval"] - B[s]["bestval"]})
        sel = [x for x in dvr if x["F"] == F]
        if sel:
            print(f"debt_vs_rr |F|={F}: routes equal {sum(x['routes_equal'] for x in sel)}/{len(sel)}, "
                  f"final ppl differs {sum(x['final_diff'] != 0 for x in sel)}/{len(sel)}, "
                  f"max |diff| {max(abs(x['final_diff']) for x in sel):.3f}")

    # ── write ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(OUTS, (table, tests, dyn, curves, summary, dvr)):
        rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        with open(out_dir / name, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
