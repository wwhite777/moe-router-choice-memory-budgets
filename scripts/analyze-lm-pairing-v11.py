"""Post hoc (2026-09-30): epoch-to-epoch expert-to-window pairing in the LM route traces.

For every LM run, routes_train.npy holds the expert chosen per training step and layer
(-1 = null path). Training batches are contiguous BPTT windows in the same order every
epoch, so step s of epoch e and step s of epoch e+1 process the same text window. This
script reports, per arm, how often a window meets the same expert in consecutive epochs
(lag-1 repeat rate) and over all epoch pairs, pooled over layers and seeds.

A cyclic schedule whose per-epoch phase drift is not a multiple of |F| rotates the
pairing (repeat rate 0); an i.i.d. random schedule repeats at rate about 1/|F|.

Usage: python scripts/analyze-lm-pairing-v11.py --logs-dir result/logs \
           --prefix e6lm3_ --out result/tables/lm_v11/lm_pairing.csv
Exits non-zero if no run is found.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs-dir", required=True)
ap.add_argument("--prefix", required=True, help="e.g. e6lm3_ or e6lm2_")
ap.add_argument("--out", required=True)
args = ap.parse_args()

runs = sorted(p for p in Path(args.logs_dir).iterdir()
              if p.name.startswith(args.prefix) and (p / "routes_train.npy").exists())
if not runs:
    sys.exit(f"no runs with prefix {args.prefix} in {args.logs_dir}")

per_arm = defaultdict(list)
for run in runs:
    cfg = json.load(open(run / "results.json"))["config"]
    n_ep = cfg["epochs"]
    r = np.load(run / "routes_train.npy").astype(np.int16)  # (steps, layers)
    if r.shape[0] % n_ep:
        sys.exit(f"{run.name}: {r.shape[0]} steps not divisible by {n_ep} epochs")
    spe = r.shape[0] // n_ep
    r = r.reshape(n_ep, spe, r.shape[1])
    routed = r >= 0
    lag1 = [(r[e] == r[e + 1])[routed[e] & routed[e + 1]] for e in range(n_ep - 1)]
    allp = [(r[a] == r[b])[routed[a] & routed[b]]
            for a in range(n_ep) for b in range(a + 1, n_ep)]
    l1 = np.concatenate(lag1)
    ap_ = np.concatenate(allp)
    per_arm[(cfg["method"], cfg["d_model"], cfg["budget"])].append(
        dict(run=run.name, seed=cfg["seed"], steps_per_epoch=spe,
             lag1=float(l1.mean()) if l1.size else float("nan"),
             allpairs=float(ap_.mean()) if ap_.size else float("nan"),
             n_lag1=int(l1.size)))

rows = []
for (method, d, budget), v in sorted(per_arm.items()):
    rows.append(dict(method=method, d_model=d, budget_bytes=budget, n_runs=len(v),
                     steps_per_epoch=";".join(sorted({str(x["steps_per_epoch"]) for x in v})),
                     lag1_repeat_mean=np.mean([x["lag1"] for x in v]),
                     lag1_repeat_min=np.min([x["lag1"] for x in v]),
                     lag1_repeat_max=np.max([x["lag1"] for x in v]),
                     allpairs_repeat_mean=np.mean([x["allpairs"] for x in v]),
                     n_routed_pairs=sum(x["n_lag1"] for x in v)))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(runs)} runs, {len(rows)} arms -> {args.out}")
