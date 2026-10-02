"""Sensitivity and context analyses requested by the external reviews of the manuscript (added 2026-10-01; all POST HOC).

Writes result/tables/v13/ (refuses to overwrite an existing file):
  seedblock5.csv        vision equivalence re-run on the first five seeds only {42, 123, 7, 2026, 31415}
                        (the protocol's screening + confirmatory blocks), same tests as analyze-equivalence.py
                        (paired TOST at eps = 0.005, Holm within (grid, router-pair) families across cells)
  seedblock5_summary.csv
  dedup_summary.csv     genuine-pair equivalence rates with and without pairs that contain Debt-Base
                        (Debt-Base duplicates round-robin by Proposition 1)
  expert_benefit.csv    mean final / best-validation accuracy by realized |F| (S2) or m (S1), round-robin and all
                        routers, and paired round-robin differences vs |F| = 0 (S2) or m = 1 (S1)
  lm_seed_modes.csv     LM per-seed best-validation epochs (1-indexed) and how many seeds peak at the last epoch
  lm_sign_tests.csv     two-sided sign and Wilcoxon tests for the LM contrasts vs round-robin (final and best-val)

Inputs: result/logs/*/results.json (s1, e1, e4), result/frozen_2026-08-24/{s1_summary,s2_summary,route_disagreement}.csv,
result/tables/equiv_categories.csv, result/tables/lm_v11/{lm_summary,lm_tests}.csv.
Usage: python scripts/analyze-review-sensitivity-v13.py [--out-dir result/tables/v13]
"""
import argparse
import csv
import json
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
TABLES = ROOT / "result" / "tables"
PAT = re.compile(r"^(s1|e1|e4)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")
EPS = 0.005
FIRST5 = {42, 123, 7, 2026, 31415}
DB = "gatedmoe_base"


def tost_p(d, eps):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < eps else 1.0
    return max(1 - stats.t.cdf((d.mean() + eps) / se, n - 1), stats.t.cdf((d.mean() - eps) / se, n - 1))


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p, kind="mergesort")):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def write(path, rows):
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    if not rows:
        sys.exit(f"no rows for {path}")
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def genuine_keys():
    dis = defaultdict(float)
    for r in csv.DictReader(open(FROZEN / "route_disagreement.csv")):
        k = (r["grid"], r["dataset"], r["cell"], frozenset((r["method_a"], r["method_b"])))
        dis[k] = max(dis[k], float(r["disagreement"]))
    return dis


def seedblock5(dis):
    cells = defaultdict(dict)
    for d in sorted(LOGS.iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        grid, ds, method, cell, seed = m.groups()
        if int(seed) not in FIRST5:
            continue
        cells[(grid, ds, cell)][(method, int(seed))] = json.load(open(d / "results.json"))["final_test_acc"]
    rows = []
    for (grid, ds, cell), data in sorted(cells.items()):
        for a, b in combinations(sorted({mm for mm, _ in data}), 2):
            seeds = sorted({s for (mm, s) in data if mm == a} & {s for (mm, s) in data if mm == b})
            if len(seeds) < 5:
                continue
            d = np.array([data[(a, s)] - data[(b, s)] for s in seeds])
            if np.allclose(d, 0):
                p_diff, p_t = 1.0, 0.0
            else:
                p_diff = stats.ttest_1samp(d, 0.0).pvalue
                p_t = tost_p(d, EPS)
            k = (grid, ds, cell, frozenset((a, b)))
            rows.append({"grid": grid, "dataset": ds, "cell": cell, "pair": f"{a}|{b}", "n_seeds": len(seeds),
                         "mean_diff": float(d.mean()), "p_diff": float(p_diff), "p_tost": float(p_t),
                         "genuine": bool(dis.get(k, 0.0) > 0), "has_debt_base": DB in (a, b)})
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r["grid"], r["pair"])].append(i)
    for idx in fam.values():
        ad, at = holm([rows[i]["p_diff"] for i in idx]), holm([rows[i]["p_tost"] for i in idx])
        for k, i in enumerate(idx):
            rows[i]["p_diff_holm"], rows[i]["p_tost_holm"] = float(ad[k]), float(at[k])
            rows[i]["category"] = ("different" if ad[k] < 0.05 else "equivalent" if at[k] < 0.05 else "unresolved")
    summ = []
    for label, sel in [("S1", lambda r: r["grid"] == "s1"), ("S2 main", lambda r: r["grid"] == "e1"),
                       ("S2 device-scale", lambda r: r["grid"] == "e4"), ("all vision", lambda r: True)]:
        g = [r for r in rows if sel(r) and r["genuine"]]
        c = Counter(r["category"] for r in g)
        summ.append({"set": label, "genuine_pairs": len(g), "equivalent": c["equivalent"], "different": c["different"],
                     "unresolved": c["unresolved"], "equivalent_rate": c["equivalent"] / len(g) if g else float("nan")})
    return rows, summ


def dedup(dis):
    out = []
    cat = list(csv.DictReader(open(TABLES / "equiv_categories.csv")))
    for label, grids in [("S1", {"s1"}), ("S2 (main + device-scale)", {"e1", "e4"})]:
        for drop in (False, True):
            c = Counter()
            for r in cat:
                a, b = r["pair"].split("|")
                if r["grid"] not in grids or dis.get((r["grid"], r["dataset"], r["cell"], frozenset((a, b))), 0) <= 0:
                    continue
                if drop and DB in (a, b):
                    continue
                c[r["category"]] += 1
            n = sum(c.values())
            out.append({"set": label, "debt_base_pairs": "removed" if drop else "kept", "genuine_pairs": n,
                        "equivalent": c["equivalent"], "unresolved": c["unresolved"], "different": c["different"],
                        "equivalent_rate": c["equivalent"] / n})
    return out


def expert_benefit():
    out = []
    s1 = list(csv.DictReader(open(FROZEN / "s1_summary.csv")))
    s2 = list(csv.DictReader(open(FROZEN / "s2_summary.csv")))
    for grid, ds, rows, key, ref in [("s1", "c100", s1, "m", "1")] + \
            [(g, d, [r for r in s2 if r["grid"] == g and r["dataset"] == d], "expected_F", "0")
             for g, d in [("e1", "c10"), ("e1", "c100"), ("e4", "c10")]]:
        rr = defaultdict(dict)
        for r in rows:
            if r["method"] == "round_robin":
                rr[r[key]].setdefault(int(r["seed"]), []).append(float(r["final_test_acc"]))
        for lev in sorted({r[key] for r in rows}, key=int):
            allm = [float(r["final_test_acc"]) for r in rows if r[key] == lev]
            allb = [float(r["test_acc_at_best_val"]) for r in rows if r[key] == lev and r.get("test_acc_at_best_val")]
            rrv = [float(r["final_test_acc"]) for r in rows if r[key] == lev and r["method"] == "round_robin"]
            # paired (by seed) round-robin difference vs the reference level; S1 averages the two masks per seed
            a = {s: np.mean(v) for s, v in rr[lev].items()}
            b = {s: np.mean(v) for s, v in rr[ref].items()}
            seeds = sorted(set(a) & set(b))
            d = np.array([a[s] - b[s] for s in seeds])
            out.append({"grid": grid, "dataset": ds, "level_name": key, "level": lev,
                        "all_routers_final_mean": float(np.mean(allm)), "all_routers_bestval_mean": float(np.mean(allb)),
                        "round_robin_final_mean": float(np.mean(rrv)), "n_round_robin_runs": len(rrv),
                        "rr_paired_diff_vs_ref": float(d.mean()) if len(d) else float("nan"),
                        "rr_paired_n": len(d), "rr_paired_seeds_lower": int((d < 0).sum()) if len(d) else 0,
                        "reference_level": ref})
    return out


def lm_modes_and_signs():
    summ = list(csv.DictReader(open(TABLES / "lm_v11" / "lm_summary.csv")))
    by = defaultdict(dict)
    for r in summ:
        by[(r["backbone"], r["F"], r["method"])][r["seed"]] = r
    modes = []
    for k, v in sorted(by.items()):
        eps = sorted(int(x["best_val_epoch"]) for x in v.values())  # lm_summary.csv is already 1-indexed
        fin = [float(x["final_test_ppl"]) for x in v.values()]
        modes.append({"backbone": k[0], "F": k[1], "method": k[2], "n": len(v),
                      "best_epochs_sorted_1idx": ";".join(map(str, eps)),
                      "seeds_best_at_last_epoch": sum(e == 10 for e in eps),
                      "final_min": min(fin), "final_max": max(fin)})
    signs = []
    for k, v in sorted(by.items()):
        rr = by.get((k[0], k[1], "round_robin"))
        if k[2] == "round_robin" or not rr:
            continue
        for ep, col in (("final", "final_test_ppl"), ("bestval", "test_ppl_at_best_val")):
            seeds = sorted(set(v) & set(rr))
            d = np.array([float(v[s][col]) - float(rr[s][col]) for s in seeds])
            nz = d[d != 0]
            pos = int((nz > 0).sum())
            sign_p = float(stats.binomtest(pos, len(nz)).pvalue) if len(nz) else 1.0
            wil_p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
            signs.append({"backbone": k[0], "F": k[1], "method": k[2], "endpoint": ep, "n": len(d),
                          "mean_diff": float(d.mean()), "nonzero": len(nz), "seeds_higher_ppl": pos,
                          "sign_p": sign_p, "wilcoxon_p": wil_p})
    return modes, signs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(TABLES / "v13"))
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dis = genuine_keys()
    rows, summ = seedblock5(dis)
    write(out / "seedblock5.csv", rows)
    write(out / "seedblock5_summary.csv", summ)
    write(out / "dedup_summary.csv", dedup(dis))
    write(out / "expert_benefit.csv", expert_benefit())
    modes, signs = lm_modes_and_signs()
    write(out / "lm_seed_modes.csv", modes)
    write(out / "lm_sign_tests.csv", signs)
    print(f"wrote 6 tables to {out}; seed-block-5 pairs {len(rows)}")
    for s in summ:
        print("seedblock5", s)


if __name__ == "__main__":
    main()
