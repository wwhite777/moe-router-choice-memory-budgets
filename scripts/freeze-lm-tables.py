"""Freeze E6-LM results into CSVs for the figure scripts.

lm_summary.csv: one row per run (method, budget_kb, F, seed, final_test_ppl,
                test_ppl_at_best_val).
lm_curves.csv:  per-epoch test ppl for every LM run (for the divergence figure).
Copied into result/frozen_2026-08-24/ alongside the other frozen tables.
"""

import sys
import json
import csv
import re
import glob
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import TABLE_DIR, LOG_DIR, RESULT_DIR

PAT = re.compile(r"e6lm2_(\w+?)_B(\d+)k_s(\d+)$")
F = {17991: 0, 24457: 4, 42487: 8}

rows, curves = [], []
for p in sorted(glob.glob(str(Path(LOG_DIR) / 'e6lm2_*' / 'results.json'))):
    m = PAT.match(Path(p).parent.name)
    if not m:
        continue
    meth, bkb, seed = m.group(1), int(m.group(2)), int(m.group(3))
    j = json.load(open(p))
    rows.append({"method": meth, "budget_kb": bkb, "F": F[bkb], "seed": seed,
                 "final_test_ppl": j["final_test_ppl"],
                 "test_ppl_at_best_val": j.get("test_ppl_at_best_val")})
    for e in j["epochs"]:
        curves.append({"method": meth, "budget_kb": bkb, "F": F[bkb],
                       "seed": seed, "epoch": e["epoch"],
                       "test_ppl": e["test_ppl"], "val_ppl": e["val_ppl"]})

for name, data in [("lm_summary.csv", rows), ("lm_curves.csv", curves)]:
    out = Path(TABLE_DIR) / name
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
    shutil.copy(out, Path(RESULT_DIR) / "frozen_2026-08-24")
    print(f"{name}: {len(data)} rows, frozen")
