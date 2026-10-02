"""Post hoc context tests (not declared in PROTOCOL.md; unadjusted p-values, labeled post hoc).

Reads <logs>/<run>/results.json and result/tables/v13/expert_benefit.csv; writes to <out-dir>
(default result/tables/v14/; refuses to overwrite):
  a5_shared_context.csv     the task-trained gated_top1 arm vs the shared path alone (round_robin at
                            realized |F|=0), paired by seed:
                              vision: e1a5_c100_gated_top1_B7333k (|F|=4) and _B17350k (|F|=8)
                                      vs e1_c100_round_robin_B3907k (|F|=0); accuracy as a fraction
                              LM:     e6lm5_gated_top1_B24457k (|F|=4) and _B42487k (|F|=8)
                                      vs e6lm3_round_robin_B17991k (|F|=0); test perplexity
                            endpoints final and bestval (test metric at the best-validation epoch);
                            mean_diff = arm minus comparator; two-sided paired t p; TOST with margin
                            0.005 (vision) or 1% of the comparator mean for that endpoint (LM): p_tost =
                            max of the two one-sided paired t p-values, df = n-1, and the 90% CI
  expert_benefit_tests.csv  for every row of expert_benefit.csv, the round-robin paired difference vs
                            the reference level recomputed from the raw runs (S1: masks averaged within
                            seed, m vs m=1; S2 grids e1/e4: runs grouped by realized |F| and averaged
                            within seed across budgets with the same |F|, paired on the seeds shared
                            with |F|=0), with seeds lower/higher and the two-sided paired t p
                            (unadjusted). The paired differences, n and seeds lower must equal
                            expert_benefit.csv (differences to 1e-12); the script fails otherwise.
Every run used must have realized |F| (feasible-set histogram) equal to the stated level.
Exit 1 (nothing written) if a run is missing or a check fails.
Usage: python3 scripts/analyze-context-v14.py [--logs DIR] [--expert-benefit FILE] [--out-dir DIR]
Dependencies: numpy, scipy (standard library otherwise).
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "result" / "logs"
EB = ROOT / "result" / "tables" / "v13" / "expert_benefit.csv"
OUT_DEFAULT = ROOT / "result" / "tables" / "v14"
OUTPUTS = ("a5_shared_context.csv", "expert_benefit_tests.csv")
SEEDS = [7, 42, 123, 777, 888, 999, 2026, 5150, 31415, 60486]
RR = "round_robin"
VIS_MARGIN = 0.005
LM_MARGIN_FRAC = 0.01
# (domain, F, arm run stem, comparator run stem, endpoint keys)
VIS_EP = {"final": "final_test_acc", "bestval": "test_acc_at_best_val"}
LM_EP = {"final": "final_test_ppl", "bestval": "test_ppl_at_best_val"}
CONTEXT = [
    ("vision", 4, "e1a5_c100_gated_top1_B7333k", "e1_c100_round_robin_B3907k", VIS_EP),
    ("vision", 8, "e1a5_c100_gated_top1_B17350k", "e1_c100_round_robin_B3907k", VIS_EP),
    ("lm", 4, "e6lm5_gated_top1_B24457k", "e6lm3_round_robin_B17991k", LM_EP),
    ("lm", 8, "e6lm5_gated_top1_B42487k", "e6lm3_round_robin_B17991k", LM_EP),
]
PAT = re.compile(r"^(s1|e1|e4)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")


def paired_p(a, b):
    """Two-sided paired t-test p; identical vectors -> 1.0 (as in reproduce_numbers_v11.py)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0) or d.std(ddof=1) == 0:
        return 1.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def tost_paired(d, margin):
    """Two one-sided paired t-tests of |mean diff| < margin; p_tost = max of the two; 90% CI."""
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    p_lower = 1 - stats.t.cdf((m + margin) / se, df)  # H0: diff <= -margin
    p_upper = stats.t.cdf((m - margin) / se, df)      # H0: diff >= +margin
    h = stats.t.ppf(0.95, df) * se
    return float(max(p_lower, p_upper)), float(m - h), float(m + h)


def realized_F(j):
    fh = np.array([e["feasible_hist"] for e in j["epochs"]]).sum(axis=0)
    return sorted(int(i) for i in np.nonzero(fh)[0])


def load(logs, name, problems):
    p = logs / name / "results.json"
    if not p.exists():
        problems.append(f"{name}: results.json missing")
        return None
    return json.load(open(p))


def shared_context(logs, problems):
    rows = []
    for dom, F, arm, comp, ep in CONTEXT:
        A, B = {}, {}
        for s in SEEDS:
            for stem, store, want in ((arm, A, F), (comp, B, 0)):
                j = load(logs, f"{stem}_s{s}", problems)
                if j is None:
                    continue
                if realized_F(j) != [want]:
                    problems.append(f"{stem}_s{s}: realized |F| {realized_F(j)} != [{want}]")
                store[s] = j
        if len(A) != len(SEEDS) or len(B) != len(SEEDS):
            continue
        for key in ("final", "bestval"):
            x = np.array([float(A[s][ep[key]]) for s in SEEDS])
            y = np.array([float(B[s][ep[key]]) for s in SEEDS])
            d = x - y
            margin = VIS_MARGIN if dom == "vision" else LM_MARGIN_FRAC * float(y.mean())
            pt, lo, hi = tost_paired(d, margin)
            rows.append({"domain": dom, "F": F, "endpoint": key, "n": len(d),
                         "mean_arm": float(x.mean()), "mean_comparator": float(y.mean()),
                         "mean_diff": float(d.mean()), "seeds_higher": int((d > 0).sum()),
                         "seeds_lower": int((d < 0).sum()), "t_p": paired_p(x, y), "margin": float(margin),
                         "tost_p": pt, "ci90_lo": lo, "ci90_hi": hi,
                         "arm_runs": f"{arm}_s*", "comparator_runs": f"{comp}_s*"})
    return rows


def expert_benefit(logs, eb_rows, problems):
    # raw round-robin runs of s1, e1, e4
    raw = defaultdict(lambda: defaultdict(list))  # (grid, dataset) -> level -> [(seed, acc)]
    for d in sorted(logs.iterdir()):
        m = PAT.match(d.name)
        if m is None or m.group(3) != RR or not d.is_dir():
            continue
        g, ds, _, cell, seed = m.groups()
        j = load(logs, d.name, problems)
        if j is None:
            continue
        if g == "s1":
            lev = re.match(r"m(\d+)v\d+$", cell).group(1)
        else:
            F = realized_F(j)
            if len(F) != 1:
                problems.append(f"{d.name}: realized |F| {F} is not a single level")
                continue
            lev = str(F[0])
        raw[(g, ds)][lev].append((int(seed), float(j["final_test_acc"])))
    rows = []
    for e in eb_rows:
        key, lev, ref = (e["grid"], e["dataset"]), e["level"], e["reference_level"]
        per = {}
        for L in (lev, ref):
            acc = defaultdict(list)
            for s, v in raw[key][L]:
                acc[s].append(v)
            per[L] = {s: float(np.mean(v)) for s, v in acc.items()}
        seeds = sorted(set(per[lev]) & set(per[ref]))
        a = np.array([per[lev][s] for s in seeds])
        b = np.array([per[ref][s] for s in seeds])
        dd = a - b
        md = float(dd.mean()) if len(dd) else float("nan")
        n_rr = len(raw[key][lev])
        exp_d, exp_n, exp_lo = float(e["rr_paired_diff_vs_ref"]), int(e["rr_paired_n"]), int(e["rr_paired_seeds_lower"])
        if not (abs(md - exp_d) <= 1e-12 and len(dd) == exp_n and int((dd < 0).sum()) == exp_lo
                and n_rr == int(e["n_round_robin_runs"])):
            problems.append(f"expert_benefit mismatch {e['grid']} {e['dataset']} {e['level_name']}={lev}: "
                            f"diff {md!r} vs {exp_d!r}, n {len(dd)} vs {exp_n}, lower {int((dd < 0).sum())} vs {exp_lo}, "
                            f"round-robin runs {n_rr} vs {e['n_round_robin_runs']}")
        rows.append({"grid": e["grid"], "dataset": e["dataset"], "level_name": e["level_name"],
                     "level": int(lev), "reference_level": int(ref), "n": len(dd), "rr_paired_diff_vs_ref": md,
                     "seeds_lower": int((dd < 0).sum()), "seeds_higher": int((dd > 0).sum()),
                     "t_p": paired_p(a, b) if len(dd) > 1 else float("nan")})
    return rows


def write(path, rows, order):
    rows = sorted(rows, key=order)
    with open(path, "x", newline="") as f:  # "x": never overwrite
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def rel(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(LOGS))
    ap.add_argument("--expert-benefit", default=str(EB))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    logs, out = Path(a.logs), Path(a.out_dir)
    clash = [f for f in OUTPUTS if (out / f).exists()]
    if clash:
        sys.exit(f"refusing to overwrite {rel(out)}: {clash}")
    problems = []
    ctx = shared_context(logs, problems)
    eb = expert_benefit(logs, list(csv.DictReader(open(a.expert_benefit))), problems)
    if len(ctx) != 2 * len(CONTEXT):
        problems.append(f"shared-path context: {len(ctx)}/{2 * len(CONTEXT)} rows")
    if problems:
        for p in problems:
            print("CHECK FAILED:", p)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    write(out / OUTPUTS[0], ctx, lambda r: (r["domain"], r["F"], r["endpoint"]))
    write(out / OUTPUTS[1], eb, lambda r: (r["grid"], r["dataset"], r["level"]))
    for r in sorted(ctx, key=lambda r: (r["domain"], r["F"], r["endpoint"])):
        print(f"context {r['domain']} |F|={r['F']} {r['endpoint']}: diff {r['mean_diff']:+.6g} "
              f"({r['seeds_higher']} higher / {r['seeds_lower']} lower of {r['n']}), t p {r['t_p']:.8g}, "
              f"TOST p {r['tost_p']:.8g} (margin {r['margin']:.9g}), 90% CI [{r['ci90_lo']:.9g}, {r['ci90_hi']:.9g}]")
    print(f"expert benefit: {len(eb)} rows; paired diffs, n and seeds lower equal expert_benefit.csv")
    for r in eb:
        print(f"  {r['grid']} {r['dataset']} {r['level_name']}={r['level']}: diff {r['rr_paired_diff_vs_ref']:+.6g}, "
              f"lower {r['seeds_lower']}/{r['n']}, t p {r['t_p']:.6g}")
    print(f"wrote {rel(out / OUTPUTS[0])} ({len(ctx)} rows), {rel(out / OUTPUTS[1])} ({len(eb)} rows)")


if __name__ == "__main__":
    main()
