"""Secondary-endpoint analysis (PROTOCOL.md §1: test metric at the best-VALIDATION epoch).

Added after the primary tables were frozen. The frozen primary analysis
(analyze-equivalence.py -> equivalence.csv, final-epoch endpoint) is untouched;
this script applies the SAME method to the pre-declared secondary endpoint.

Vision (s1, e1, e4): paired per-seed differences of test_acc_at_best_val for every
method pair in every matched cell (n >= 5 seeds); two-sided paired t-test; TOST at
eps = 0.005 (sensitivity 0.5x, 2x); Holm within each (grid, pair) family, separately
for difference tests and TOST -- identical to analyze-equivalence.py.
Each pair is also classed by its route traces, read from the frozen per-seed route-disagreement
table (result/frozen_2026-08-24/route_disagreement.csv, computed from routes_train.npy):
  tied    = identical routes in every seed AND identical training (sd of diffs = 0 or
            no learned router involved),
  auxonly = identical routes, learned router involved (differs only through its
            auxiliary loss),
  choice  = routes differ in at least one seed.
LM (e6lm2, main backbone): paired per-seed differences vs round-robin of final test
perplexity (primary) and test perplexity at the best-validation epoch (secondary);
Holm over the 12-test family {Debt-Base, Debt-Grad, Random, Learned} x |F| in {0,4,8}
(the Table 1 family); the other arms get unadjusted paired p-values.

Writes result/tables/equivalence_secondary.csv and result/tables/lm_secondary.csv.
"""

import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "result" / "logs"
TABLE_DIR = ROOT / "result" / "tables"
if len(sys.argv) > 1:  # optional output directory (e.g. for regeneration checks)
    TABLE_DIR = Path(sys.argv[1])
ROUTE_TABLE = ROOT / "result" / "frozen_2026-08-24" / "route_disagreement.csv"

PAT = re.compile(r"(s1|e1|e4)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")
LM_PAT = re.compile(r"e6lm2_(\w+?)_B(\d+)k_s(\d+)$")
EPS = 0.005
MIN_SEEDS = 5
LM_F = {17991: 0, 24457: 4, 42487: 8}
LM_FAMILY = ("gatedmoe_base", "gatedmoe_g", "random", "learned_top1")


def tost_p(diffs, eps):
    n = len(diffs)
    se = diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(diffs.mean()) < eps else 1.0
    p_lo = 1 - stats.t.cdf((diffs.mean() + eps) / se, n - 1)
    p_hi = stats.t.cdf((diffs.mean() - eps) / se, n - 1)
    return max(p_lo, p_hi)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def vision():
    acc = defaultdict(dict)
    maxdis = defaultdict(float)
    with open(ROUTE_TABLE) as f:
        for r in csv.DictReader(f):
            k = (r["grid"], r["dataset"], r["cell"],
                 "|".join(sorted((r["method_a"], r["method_b"]))))
            maxdis[k] = max(maxdis[k], float(r["disagreement"]))
    for d in sorted(LOG_DIR.iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        grid, ds, method, cell, seed = m.groups()
        j = json.load(open(d / "results.json"))
        acc[(grid, ds, cell)][(method, int(seed))] = (
            j["final_test_acc"], j["test_acc_at_best_val"])

    rows = []
    for key, data in sorted(acc.items()):
        grid, ds, cell = key
        methods = sorted({m for m, _ in data})
        for a, b in combinations(methods, 2):
            seeds = sorted({s for (m, s) in data if m == a} &
                           {s for (m, s) in data if m == b})
            if len(seeds) < MIN_SEEDS:
                continue
            same_routes = maxdis[(grid, ds, cell, f"{a}|{b}")] == 0.0
            row = {"grid": grid, "dataset": ds, "cell": cell,
                   "n_seeds": len(seeds), "pair": f"{a}|{b}"}
            for ep_i, ep in enumerate(("final", "bestval")):
                diffs = np.array([data[(a, s)][ep_i] - data[(b, s)][ep_i]
                                  for s in seeds])
                if np.allclose(diffs, 0):
                    p_diff, p_tost = 1.0, 0.0
                else:
                    p_diff = stats.ttest_rel(
                        [data[(a, s)][ep_i] for s in seeds],
                        [data[(b, s)][ep_i] for s in seeds]).pvalue
                    p_tost = tost_p(diffs, EPS)
                row[f"{ep}_mean_diff"] = round(float(diffs.mean()), 5)
                row[f"{ep}_sd_diff"] = round(float(diffs.std(ddof=1)), 5)
                row[f"{ep}_p_diff"] = float(p_diff)
                row[f"{ep}_p_tost"] = float(p_tost)
                row[f"{ep}_p_tost_half"] = float(tost_p(diffs, EPS / 2))
                row[f"{ep}_p_tost_2x"] = float(tost_p(diffs, EPS * 2))
            if not same_routes:
                row["cls"] = "choice"
            elif "learned_top1" in row["pair"] and row["final_sd_diff"] > 0:
                row["cls"] = "auxonly"
            else:
                row["cls"] = "tied"
            rows.append(row)

    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r["grid"], r["pair"])].append(i)
    for idxs in fam.values():
        for ep in ("final", "bestval"):
            adj_d = holm([rows[i][f"{ep}_p_diff"] for i in idxs])
            adj_t = holm([rows[i][f"{ep}_p_tost"] for i in idxs])
            for k, i in enumerate(idxs):
                rows[i][f"{ep}_p_diff_holm"] = float(adj_d[k])
                rows[i][f"{ep}_p_tost_holm"] = float(adj_t[k])
                rows[i][f"{ep}_equivalent"] = bool(adj_t[k] < 0.05)
                rows[i][f"{ep}_sig_diff"] = bool(adj_d[k] < 0.05)

    out = TABLE_DIR / "equivalence_secondary.csv"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def tally(sel, label):
        n = len(sel)
        for ep in ("final", "bestval"):
            eq = sum(r[f"{ep}_equivalent"] for r in sel)
            sig = sum(r[f"{ep}_sig_diff"] for r in sel)
            print(f"  {label:14s} {ep:8s} n={n:3d} equivalent={eq:3d} "
                  f"({100 * eq / n:5.1f}%) sig_diff={sig} unresolved={n - eq - sig}")
    print(f"vision: {len(rows)} cell-pairs -> {out}")
    for g in ("all", "s1", "e1", "e4"):
        sel = rows if g == "all" else [r for r in rows if r["grid"] == g]
        tally(sel, f"{g}/all")
        tally([r for r in sel if r["cls"] == "choice"], f"{g}/choice")
    for r in rows:
        if r["bestval_sig_diff"]:
            print("  SIG-DIFF (bestval):", r["grid"], r["dataset"], r["cell"],
                  r["pair"], r["bestval_mean_diff"], round(r["bestval_p_diff_holm"], 4))
    return rows


def lm():
    data = defaultdict(dict)
    for d in sorted(LOG_DIR.iterdir()):
        m = LM_PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        method, budget, seed = m.group(1), int(m.group(2)), int(m.group(3))
        j = json.load(open(d / "results.json"))
        data[(LM_F[budget], method)][seed] = (
            j["final_test_ppl"], j["test_ppl_at_best_val"], j["best_val_epoch"],
            j["epochs"][-1]["train_ppl"])
    rows = []
    for (F, method), per_seed in sorted(data.items()):
        rr = data.get((F, "round_robin"), {})
        vals = np.array(list(per_seed.values()))
        row = {"F": F, "method": method, "n": len(per_seed),
               "final_mean": round(float(vals[:, 0].mean()), 2),
               "final_sd": round(float(vals[:, 0].std(ddof=1)), 2),
               "bestval_mean": round(float(vals[:, 1].mean()), 2),
               "bestval_sd": round(float(vals[:, 1].std(ddof=1)), 2),
               "best_val_epoch_mean_1idx": round(float(vals[:, 2].mean()) + 1, 2),
               "final_train_ppl_mean": round(float(vals[:, 3].mean()), 2)}
        if method != "round_robin":
            seeds = sorted(set(per_seed) & set(rr))
            for ep_i, ep in enumerate(("final", "bestval")):
                a = np.array([per_seed[s][ep_i] for s in seeds])
                b = np.array([rr[s][ep_i] for s in seeds])
                row[f"{ep}_diff_vs_rr"] = round(float((a - b).mean()), 2)
                row[f"{ep}_pos_seeds"] = int(((a - b) > 0).sum())
                row[f"{ep}_p_vs_rr"] = float(
                    stats.ttest_rel(a, b).pvalue) if not np.allclose(a, b) else 1.0
        rows.append(row)
    for ep in ("final", "bestval"):
        fam = [i for i, r in enumerate(rows) if r["method"] in LM_FAMILY]
        adj = holm([rows[i][f"{ep}_p_vs_rr"] for i in fam])
        for k, i in enumerate(fam):
            rows[i][f"{ep}_p_holm12"] = float(adj[k])
    out = TABLE_DIR / "lm_secondary.csv"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"lm: {len(rows)} (|F|, arm) rows -> {out}")
    for r in rows:
        print("  |F|={F} {method:22s} final {final_mean:6.1f}±{final_sd:4.1f} "
              "bestval {bestval_mean:6.1f}±{bestval_sd:3.1f} best-val epoch {best_val_epoch_mean_1idx:4.1f} "
              "train_ppl {final_train_ppl_mean:5.1f}".format(**r),
              "" if r["method"] == "round_robin" else
              "| vs RR: final {final_diff_vs_rr:+6.1f} ({final_pos_seeds}/{n}, p={final_p_vs_rr:.2g}) "
              "bestval {bestval_diff_vs_rr:+5.1f} ({bestval_pos_seeds}/{n}, p={bestval_p_vs_rr:.2g})".format(**r)
              + ("" if "bestval_p_holm12" not in r else
                 " holm12: final {final_p_holm12:.2g} bestval {bestval_p_holm12:.2g}".format(**r)))


if __name__ == "__main__":
    vision()
    lm()
