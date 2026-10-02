"""Small descriptive quantities quoted in the manuscript that no other script writes.

Since Amendment 4 (2026-09-30, PROTOCOL.md): the LM numbers come from the corrected
rerun e6lm3_* / e6lm3d256_*; the superseded e6lm2_* / e6lmd256_* runs are reported
only under ids ending in _superseded. Reads result/tables/equivalence_secondary.csv
(from analyze-secondary-endpoint.py), per-run results.json and the LM route traces.
Writes result/tables/extras_v11.csv (refuses to overwrite).
  1. Direction of the learned router among genuine vision pairs (all, and unresolved at the primary endpoint).
  2. Median wall-clock minutes per run, by run-name prefix (sum of per-epoch time_s):
     paper prefixes s1, e1, e4, e5, e6lm3, e6lm3d256; superseded e6lm2, e6lmd256 separately.
  3. Inter-activation gap values of round-robin and fixed-cadence random at |F|=4 (LM), pooled over seeds:
     corrected e6lm3 (paper) and superseded e6lm2 (separate ids).
"""

import sys
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "result" / "logs"
TABLE_DIR = ROOT / "result" / "tables"
OUT = TABLE_DIR / "extras_v11.csv"
LEARNED = "learned_top1"
PAPER_PREFIXES = {"s1", "e1", "e4", "e5", "e6lm3", "e6lm3d256"}
SUPERSEDED_PREFIXES = {"e6lm2", "e6lmd256"}
LM_MAIN, LM_SUPERSEDED = "e6lm3", "e6lm2"


def learned_direction(rows, label):
    out = []
    sel = [r for r in rows if LEARNED in r["pair"].split("|")]
    signed = []
    for r in sel:
        a = r["pair"].split("|")[0]
        d = float(r["final_mean_diff"])
        signed.append(d if a == LEARNED else -d)
    signed = np.array(signed)
    better = int((signed > 0).sum())
    p = binomtest(better, len(signed), 0.5).pvalue
    out.append((f"{label}_pairs_with_learned", len(signed), "genuine pairs that include learned_top1"))
    out.append((f"{label}_learned_higher_mean", better, "pairs in which learned_top1 has the higher mean"))
    out.append((f"{label}_binomial_p", p, "two-sided binomial test of the count against 0.5"))
    out.append((f"{label}_mean_signed_diff_pp", 100 * signed.mean(), "mean(learned - other), percentage points"))
    out.append((f"{label}_max_signed_diff_pp", 100 * signed.max(), "max(learned - other), percentage points"))
    return out


def gaps(path):
    r = np.load(path)
    vals = Counter()
    for layer in range(r.shape[1]):
        col = r[:, layer]
        for e in np.unique(col[col >= 0]):
            idx = np.where(col == e)[0]
            vals.update(np.diff(idx).tolist())
    return vals


def main():
    if OUT.exists():
        sys.exit(f"refusing to overwrite {OUT}")
    rows = list(csv.DictReader(open(TABLE_DIR / "equivalence_secondary.csv")))
    genuine = [r for r in rows if r["cls"] == "choice"]
    unresolved = [r for r in genuine
                  if r["final_equivalent"] == "False" and r["final_sig_diff"] == "False"]
    out = learned_direction(genuine, "genuine") + learned_direction(unresolved, "unresolved")

    minutes = {}
    for d in sorted(LOG_DIR.iterdir()):
        f = d / "results.json"
        if not f.exists():
            continue
        pref = d.name.split("_")[0]
        if pref not in PAPER_PREFIXES | SUPERSEDED_PREFIXES:
            continue
        ep = json.load(open(f))["epochs"]
        if isinstance(ep, list) and ep and "time_s" in ep[0]:
            minutes.setdefault(pref, []).append(sum(e["time_s"] for e in ep) / 60)
    missing = (PAPER_PREFIXES | SUPERSEDED_PREFIXES) - set(minutes)
    if missing:
        sys.exit(f"no timed runs for prefixes {sorted(missing)}")
    for pref, v in sorted(minutes.items()):
        if pref in PAPER_PREFIXES:
            out.append((f"median_minutes_{pref}", float(np.median(v)), f"median over {len(v)} runs"))
    for pref, v in sorted(minutes.items()):
        if pref in SUPERSEDED_PREFIXES:
            out.append((f"median_minutes_{pref}_superseded", float(np.median(v)),
                        f"median over {len(v)} runs; superseded grid (Amendment 4), not a paper number"))

    for grid, suffix, note in ((LM_MAIN, "", "corrected grid e6lm3"),
                               (LM_SUPERSEDED, "_superseded_e6lm2", "superseded grid e6lm2, not a paper number")):
        for arm in ("round_robin", "fixed_cadence_random"):
            files = sorted(LOG_DIR.glob(f"{grid}_{arm}_B24457k_s*/routes_train.npy"))
            if not files:
                sys.exit(f"no route traces for {grid}_{arm}_B24457k_s*")
            pooled = Counter()
            for f in files:
                pooled.update(gaps(f))
            out.append((f"gap_values_F4_{arm}{suffix}", json.dumps(dict(sorted(pooled.items()))),
                        f"pooled counts of inter-activation gaps (steps), |F|=4, all seeds and layers; "
                        f"{len(files)} runs, {sum(pooled.values())} gaps; {note}"))

    with open(OUT, "x", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "value", "definition"])
        for row in out:
            w.writerow(row)
    for row in out:
        print(row[0], row[1])
    print(f"{len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
