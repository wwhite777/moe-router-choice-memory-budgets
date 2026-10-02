"""Route-disagreement analysis (PROTOCOL.md §6).

For each matched cell (same grid/dataset/budget-or-mask/seed), compute the
pairwise route-disagreement rate between methods: the fraction of
(step, layer) decisions where the chosen expert differs (null path = -1
counts as a decision). Pairs with zero disagreement are algebraically tied;
performance gaps are only attributable to routing where disagreement is
substantial. Writes result/tables/route_disagreement.csv.
"""

import sys
import json
import csv
import re
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR, TABLE_DIR

PAT = re.compile(r"(s1|e1|e4)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")


def main():
    cells = defaultdict(dict)
    accs = {}
    for d in sorted(Path(LOG_DIR).iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        if not (d / "routes_train.npy").exists():
            continue
        grid, ds, method, cell, seed = m.groups()
        cells[(grid, ds, cell, seed)][method] = d
        accs[d.name] = json.load(open(d / "results.json"))["final_test_acc"]

    rows = []
    for key, methods in sorted(cells.items()):
        if len(methods) < 2:
            continue
        grid, ds, cell, seed = key
        routes = {meth: np.load(p / "routes_train.npy") for meth, p in methods.items()}
        for a, b in itertools.combinations(sorted(routes), 2):
            ra, rb = routes[a], routes[b]
            if ra.shape != rb.shape:
                continue
            dis = float((ra != rb).mean())
            rows.append({
                "grid": grid, "dataset": ds, "cell": cell, "seed": seed,
                "method_a": a, "method_b": b,
                "disagreement": round(dis, 4),
                "acc_delta": round(accs[methods[a].name] - accs[methods[b].name], 4),
            })
    if not rows:
        print("no matched route traces yet")
        return
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(TABLE_DIR) / "route_disagreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} matched pairs -> route_disagreement.csv")

    # Aggregate: mean disagreement and mean |acc delta| per (grid, pair)
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        k = (r["grid"], r["method_a"] + " vs " + r["method_b"])
        agg[k][0] += r["disagreement"]
        agg[k][1] += abs(r["acc_delta"])
        agg[k][2] += 1
    print(f"{'grid':>4} {'pair':>32} {'mean_disagree':>14} {'mean|d_acc|':>12} {'n':>4}")
    for (grid, pair), (sd, sa, n) in sorted(agg.items()):
        print(f"{grid:>4} {pair:>32} {sd/n:>14.4f} {sa/n:>12.4f} {n:>4}")


if __name__ == "__main__":
    main()
