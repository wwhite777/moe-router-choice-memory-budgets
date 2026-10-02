"""S2/E4 aggregation: method spread vs realized |F| on the natural budget grids.

Screening view (PROTOCOL.md §4): per (grid, dataset, budget, seed) matched cell,
spread of final-epoch test accuracy across methods; grouped by the cell's
expected |F|. Writes result/tables/s2_summary.csv. Interpretation rule: these
cells confound selection freedom with capacity — causal claims live in S1.
"""

import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR, TABLE_DIR, CONFIG_DIR

PAT = re.compile(r"(e1|e4)_(c10|c100)_(\w+)_B(\d+)k_s(\d+)$")
METHODS = ["round_robin", "gatedmoe_base", "gatedmoe_g", "learned_top1", "random"]


def expected_fsize_map():
    out = {}
    for grid in ("e1", "e4"):
        for j in json.load(open(CONFIG_DIR / f"{grid}_grid.json"))["jobs"]:
            out[(grid, j["budget"] // 1024)] = j["expected_fsize"]
    return out


def main():
    fmap = expected_fsize_map()
    rows = []
    for d in sorted(Path(LOG_DIR).iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        grid, ds, method, bkb, seed = (m.group(1), m.group(2), m.group(3),
                                       int(m.group(4)), int(m.group(5)))
        j = json.load(open(d / "results.json"))
        e = j["epochs"][-1]
        rows.append({
            "grid": grid, "dataset": ds, "method": method, "budget_kb": bkb,
            "seed": seed, "expected_F": fmap.get((grid, bkb)),
            "final_test_acc": j["final_test_acc"],
            "mean_feasible": round(e.get("mean_feasible", -1), 3),
            "peak_sram_kb": e["peak_sram_bytes"] // 1024,
            "budget_ok": e["peak_sram_bytes"] <= j["config"]["budget_bytes"],
        })
    if not rows:
        print("no completed e1/e4 runs yet")
        return
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(TABLE_DIR) / "s2_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad_budget = [r for r in rows if not r["budget_ok"]]
    bad_fsize = [r for r in rows if r["expected_F"] is not None
                 and abs(r["mean_feasible"] - r["expected_F"]) > 0.01]
    print(f"{len(rows)} runs; budget violations: {len(bad_budget)}; "
          f"|F| mismatches vs grid: {len(bad_fsize)}")
    for r in (bad_budget + bad_fsize)[:10]:
        print("  CHECK:", r)

    cells = defaultdict(dict)
    for r in rows:
        cells[(r["grid"], r["dataset"], r["budget_kb"], r["seed"])][r["method"]] = \
            r["final_test_acc"]
    by_f = defaultdict(list)
    tie_rr_base = defaultdict(list)
    for key in sorted(cells):
        accs = cells[key]
        if len(accs) < len(METHODS):
            continue
        grid, ds, bkb, seed = key
        F = fmap.get((grid, bkb))
        vals = [accs[m] for m in METHODS]
        by_f[(grid, ds, F)].append(max(vals) - min(vals))
        tie_rr_base[(grid, F)].append(
            abs(accs["round_robin"] - accs["gatedmoe_base"]))
    print("\nmean spread across 5 methods by expected |F| "
          "(matched complete cells; screening only):")
    for (grid, ds, F) in sorted(by_f, key=lambda x: (x[0], x[1], x[2] if x[2] is not None else -1)):
        sp = by_f[(grid, ds, F)]
        print(f"  {grid} {ds:>4} |F|={F}: mean={sum(sp)/len(sp):.4f} "
              f"max={max(sp):.4f} n={len(sp)}")
    print("\n|RR - DebtBase| by (grid, |F|):")
    for k in sorted(tie_rr_base, key=lambda x: (x[0], x[1] if x[1] is not None else -1)):
        v = tie_rr_base[k]
        print(f"  {k}: mean={sum(v)/len(v):.4f} max={max(v):.4f} n={len(v)}")


if __name__ == "__main__":
    main()
