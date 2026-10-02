"""Amendment-3 analyses that need no new runs (PROTOCOL A3.4-A3.6).

1. Gap-CV irregularity (A3.4): per-run mean coefficient of variation of
   per-expert inter-activation gaps from routes_train.npy, LM |F|=8 arms.
   The dose-ordering claim binds to this measured statistic.
2. S1 subset-as-random-factor (A3.6): compare subset-effect vs router-effect
   magnitudes per m from the existing two-subsets-per-size design.
3. Three-category equivalence classification: equivalent / different /
   unresolved per cell-pair, with unresolved diagnosis (near-margin mean vs
   wide CI). Writes result/tables/equiv_categories.csv.
"""

import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "result" / "logs"
TABLES = ROOT / "result" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ── 1. gap-CV irregularity (LM |F|=8) ────────────────────────────────────────
print("── A3.4 measured irregularity: inter-activation-gap CV vs final ppl (LM |F|=8) ──")
arms = defaultdict(lambda: {"cv": [], "ppl": []})
for p in sorted(glob.glob(str(LOGS / "e6lm2_*_B42487k_s*"))):
    name = Path(p).name
    m = re.match(r"e6lm2_(\w+?)_B42487k_s(\d+)$", name)
    if not m or not (Path(p) / "routes_train.npy").exists():
        continue
    meth = m.group(1)
    routes = np.load(Path(p) / "routes_train.npy")  # (steps, layers)
    cvs = []
    for l in range(routes.shape[1]):
        for e in range(8):
            idx = np.where(routes[:, l] == e)[0]
            if len(idx) > 2:
                gaps = np.diff(idx)
                if gaps.mean() > 0:
                    cvs.append(gaps.std() / gaps.mean())
    j = json.load(open(Path(p) / "results.json"))
    arms[meth]["cv"].append(float(np.mean(cvs)))
    arms[meth]["ppl"].append(j["final_test_ppl"])
rows = []
for meth, d in sorted(arms.items(), key=lambda kv: np.mean(kv[1]["cv"])):
    print(f"  {meth:>22}: gap-CV={np.mean(d['cv']):.3f}±{np.std(d['cv']):.3f} "
          f"ppl={np.mean(d['ppl']):.1f}±{np.std(d['ppl']):.1f} (n={len(d['ppl'])})")
    rows.append({"method": meth, "gap_cv": np.mean(d["cv"]),
                 "ppl": np.mean(d["ppl"]), "n": len(d["ppl"])})
if len(rows) >= 3:
    r = np.corrcoef([x["gap_cv"] for x in rows], [x["ppl"] for x in rows])[0, 1]
    print(f"  arm-level Pearson r(gap-CV, ppl) = {r:.3f} over {len(rows)} arms "
          f"(few points: report as descriptive)")

# ── 2. S1 subset factor ──────────────────────────────────────────────────────
print("\n── A3.6 S1 subset-effect vs router-effect (existing data) ──")
s1 = defaultdict(dict)  # (m, variant, seed) -> method -> acc
PAT = re.compile(r"s1_c100_(\w+?)_m(\d)v(\d)_s(\d+)$")
for p in sorted(glob.glob(str(LOGS / "s1_c100_*"))):
    m = PAT.match(Path(p).name)
    if not m or not (Path(p) / "results.json").exists():
        continue
    meth, mm, vv, seed = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    s1[(mm, vv, seed)][meth] = json.load(open(Path(p) / "results.json"))["final_test_acc"]
methods = ["round_robin", "gatedmoe_base", "learned_top1", "random"]
for mm in (2, 3, 4, 6):
    subset_effects, router_effects = [], []
    seeds = sorted({s for (m2, v, s) in s1 if m2 == mm})
    for seed in seeds:
        a0, a1 = s1.get((mm, 0, seed), {}), s1.get((mm, 1, seed), {})
        for meth in methods:
            if meth in a0 and meth in a1:
                subset_effects.append(abs(a0[meth] - a1[meth]))
        for cell in (a0, a1):
            vals = [cell[meth] for meth in methods if meth in cell]
            if len(vals) == len(methods):
                router_effects.append(max(vals) - min(vals))
    print(f"  m={mm}: mean|subset effect|={np.mean(subset_effects):.4f} "
          f"(same router, different subset)  vs  mean router spread="
          f"{np.mean(router_effects):.4f} (same subset)")

# ── 3. three-category classification ─────────────────────────────────────────
print("\n── Three-category equivalence classification (vision, n>=10) ──")
rows = list(csv.DictReader(open(ROOT / "result" / "frozen_2026-08-24" / "equivalence.csv")))
out = []
cat_count = defaultdict(lambda: defaultdict(int))
MARGIN = 0.005
for r in rows:
    eq = r["equivalent_at_eps"] == "True"
    diff = r["significant_diff"] == "True"
    if eq and not diff:
        cat = "equivalent"
    elif diff and not eq:
        cat = "different"
    elif eq and diff:
        cat = "both(trivial-diff<margin)"
    else:
        cat = "unresolved"
    mean_d, sd = abs(float(r["mean_diff"])), float(r["sd_diff"])
    n = int(r["n_seeds"])
    ci_half = 2.262 * sd / np.sqrt(n) if n == 10 else 2.093 * sd / np.sqrt(n)
    reason = ""
    if cat == "unresolved":
        reason = "near-margin-mean" if mean_d > MARGIN / 2 else "wide-CI"
    cat_count[r["grid"]][cat] += 1
    out.append({**{k: r[k] for k in ("grid", "dataset", "cell", "pair", "n_seeds",
                                     "mean_diff", "sd_diff")},
                "category": cat, "unresolved_reason": reason,
                "ci_halfwidth": round(ci_half, 5)})
with open(TABLES / "equiv_categories.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
tot = defaultdict(int)
for g, cats in sorted(cat_count.items()):
    n = sum(cats.values())
    print(f"  {g}: " + " ".join(f"{c}={v}({v/n:.0%})" for c, v in sorted(cats.items())))
    for c, v in cats.items():
        tot[c] += v
n = sum(tot.values())
print(f"  ALL: " + " ".join(f"{c}={v}({v/n:.0%})" for c, v in sorted(tot.items())))
unres = [o for o in out if o["category"] == "unresolved"]
nm = sum(1 for o in unres if o["unresolved_reason"] == "near-margin-mean")
print(f"  unresolved diagnosis: {nm} near-margin-mean, {len(unres)-nm} wide-CI")
