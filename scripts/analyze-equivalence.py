"""Confirmatory statistics (PROTOCOL.md §2): paired differences, Holm, TOST.

For every matched cell with >= MIN_SEEDS seeds and every method pair:
  - paired per-seed differences of final-epoch test accuracy (PRIMARY endpoint)
  - two-sided paired t-test p-value (difference test)
  - TOST equivalence test at the pre-declared margin (CIFAR eps=0.005),
    with sensitivity margins 0.5x and 2x
  - Holm-Bonferroni correction across cells WITHIN each (grid, pair) family,
    separately for difference tests and TOST.
Language rule: TOST pass => "practically equivalent (TOST, eps)"; TOST fail =>
"not shown equivalent". A significant difference test after Holm is a
candidate threshold effect (report; boundary seeds must replicate it).
Writes result/tables/equivalence.csv.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR, TABLE_DIR

PAT = re.compile(r"(s1|e1|e4)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")
EPS = 0.005
MIN_SEEDS = 5


def tost_p(diffs: np.ndarray, eps: float) -> float:
    """Two one-sided paired t-tests vs +/-eps; TOST p = max of the two."""
    n = len(diffs)
    se = diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(diffs.mean()) < eps else 1.0
    t_lo = (diffs.mean() + eps) / se   # H0: mean <= -eps
    t_hi = (diffs.mean() - eps) / se   # H0: mean >= +eps
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return max(p_lo, p_hi)


def holm(pvals):
    """Return Holm-adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    cells = defaultdict(dict)  # (grid, ds, cell) -> {(method, seed): acc}
    for d in sorted(Path(LOG_DIR).iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        grid, ds, method, cell, seed = m.groups()
        j = json.load(open(d / "results.json"))
        cells[(grid, ds, cell)][(method, int(seed))] = j["final_test_acc"]

    rows = []
    for key, data in sorted(cells.items()):
        grid, ds, cell = key
        methods = sorted({m for m, _ in data})
        for a, b in combinations(methods, 2):
            seeds = sorted({s for (m, s) in data if m == a} &
                           {s for (m, s) in data if m == b})
            if len(seeds) < MIN_SEEDS:
                continue
            diffs = np.array([data[(a, s)] - data[(b, s)] for s in seeds])
            if np.allclose(diffs, 0):
                p_diff, p_tost = 1.0, 0.0  # identical trajectories
            else:
                p_diff = stats.ttest_rel(
                    [data[(a, s)] for s in seeds],
                    [data[(b, s)] for s in seeds]).pvalue
                p_tost = tost_p(diffs, EPS)
            rows.append({
                "grid": grid, "dataset": ds, "cell": cell, "n_seeds": len(seeds),
                "pair": f"{a}|{b}", "mean_diff": round(float(diffs.mean()), 5),
                "sd_diff": round(float(diffs.std(ddof=1)), 5),
                "p_diff": float(p_diff), "p_tost": float(p_tost),
                "p_tost_half_eps": float(tost_p(diffs, EPS / 2)),
                "p_tost_2eps": float(tost_p(diffs, EPS * 2)),
            })

    # Holm within (grid, pair) family
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r["grid"], r["pair"])].append(i)
    for _, idxs in fam.items():
        adj_d = holm([rows[i]["p_diff"] for i in idxs])
        adj_t = holm([rows[i]["p_tost"] for i in idxs])
        for k, i in enumerate(idxs):
            rows[i]["p_diff_holm"] = float(adj_d[k])
            rows[i]["p_tost_holm"] = float(adj_t[k])
            rows[i]["equivalent_at_eps"] = bool(adj_t[k] < 0.05)
            rows[i]["significant_diff"] = bool(adj_d[k] < 0.05)

    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(TABLE_DIR) / "equivalence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    eq = sum(r["equivalent_at_eps"] for r in rows)
    sig = [r for r in rows if r["significant_diff"]]
    print(f"{n} cell-pairs with n>={MIN_SEEDS} -> equivalence.csv")
    print(f"TOST-equivalent at eps={EPS} (Holm): {eq}/{n} = {eq / n:.1%}")
    print(f"Holm-significant DIFFERENCES: {len(sig)}")
    for r in sig:
        print("  SIG-DIFF:", r["grid"], r["dataset"], r["cell"], r["pair"],
              "mean_diff=", r["mean_diff"], "p_holm=", round(r["p_diff_holm"], 4))
    # Summary by grid
    by_grid = defaultdict(lambda: [0, 0])
    for r in rows:
        by_grid[r["grid"]][0] += r["equivalent_at_eps"]
        by_grid[r["grid"]][1] += 1
    for g, (e, t) in sorted(by_grid.items()):
        print(f"  {g}: TOST-equivalent {e}/{t} = {e / t:.1%}")


if __name__ == "__main__":
    main()
