"""S1 aggregation: method spread vs choice-set size m (PROTOCOL.md §3).

Aggregates final-epoch test accuracy (PRIMARY endpoint) per (m, variant,
method, seed) from completed s1_* runs; prints the per-cell method spread and
writes result/tables/s1_summary.csv. Spread = max-min across methods within a
(m, variant, seed) triple (matched comparison). No significance claims here —
screening only; TOST happens at confirmatory time.
"""

import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR, TABLE_DIR

PAT = re.compile(r"s1_c100_(\w+)_m(\d+)v(\d+)_s(\d+)$")


def main():
    rows = []
    for d in sorted(Path(LOG_DIR).iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "results.json").exists():
            continue
        method, msize, variant, seed = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        j = json.load(open(d / "results.json"))
        rows.append({
            "method": method, "m": msize, "variant": variant, "seed": seed,
            "final_test_acc": j["final_test_acc"],
            "final_test_loss": j.get("final_test_loss"),
            "test_acc_at_best_val": j.get("test_acc_at_best_val"),
            "mask": str(j["config"].get("admissible_mask")),
        })
    if not rows:
        print("no completed s1 runs yet")
        return

    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(TABLE_DIR) / "s1_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-(m, variant, seed): spread across methods (matched comparison)
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["m"], r["variant"], r["seed"])][r["method"]] = r["final_test_acc"]

    print(f"{len(rows)} runs -> {out}")
    print(f"{'m':>2} {'v':>2} {'seed':>6} | " +
          " ".join(f"{meth:>13}" for meth in
                   ["round_robin", "gatedmoe_base", "learned_top1", "random"]) +
          " |  spread")
    by_m = defaultdict(list)
    for (msize, variant, seed) in sorted(cells):
        accs = cells[(msize, variant, seed)]
        if len(accs) < 4:
            continue  # incomplete matched cell
        vals = [accs.get(meth) for meth in
                ["round_robin", "gatedmoe_base", "learned_top1", "random"]]
        spread = max(vals) - min(vals)
        by_m[msize].append(spread)
        print(f"{msize:>2} {variant:>2} {seed:>6} | " +
              " ".join(f"{v:>13.4f}" for v in vals) + f" | {spread:>7.4f}")
    print("\nmean spread by m (matched cells, screening only — NOT significance):")
    for msize in sorted(by_m):
        sp = by_m[msize]
        print(f"  m={msize}: mean={sum(sp)/len(sp):.4f}  max={max(sp):.4f}  n_cells={len(sp)}")


if __name__ == "__main__":
    main()
