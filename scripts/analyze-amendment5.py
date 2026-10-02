"""Amendment 5 analysis (PROTOCOL.md, 2026-10-01): gated_top1 (LM + vision), dense ceiling, build check.

Reads only per-run files <logs>/<run>/results.json (and routes_train.npy for the LM gated arm) and
writes four tables to <out> (default result/tables/a5/):
  a5_runs.csv        one row per run used: role, arm, cell, seed, final and best-val endpoint,
                     best_val_epoch (1-indexed), max peak SRAM vs budget
  a5_tests.csv       per family and endpoint: paired-by-seed contrasts (mean diff = A5 arm minus
                     comparator; seeds with diff > 0; two-sided paired t p; Holm within the family
                     and endpoint; two-sided sign test p), and TOST of gated_top1 vs round_robin
  a5_buildcheck.csv  e1a5chk_* vs the original e1_* run: exact equality (==) of final_test_acc and
                     test_acc_at_best_val
  a5_diagnostics.csv LM gated arm: best-val epoch per seed and number of seeds whose best is the
                     last epoch; gap-CV from routes_train.npy (definition of reproduce_numbers_v11.py
                     gap_cv); minimum per-layer utilization entropy from the last epoch's debt_stats
Families (Holm separately per family and per endpoint):
  LM:     gated_top1 vs round_robin, vs learned_top1, at B24457k (|F|=4) and B42487k (|F|=8);
          comparators e6lm3_*; TOST margin 1% of the round_robin cell mean (same endpoint)
  vision: the same four contrasts at B7333k (|F|=4) and B17350k (|F|=8), CIFAR-100;
          comparators e1_c100_*; TOST margin 0.005 accuracy
  dense:  dense (B17350k config) vs round_robin at B3907k (|F|=0) and at B17350k (|F|=8)
Direction: LM metric = perplexity (lower is better); vision = accuracy (higher is better).
Per Amendment 5, if the build check is not exact the vision and dense contrasts against the
e1_ comparators are NOT the declared analysis (column comparator_valid = no).

Refuses to overwrite any output; exits 1 if any expected run is missing unless --allow-incomplete
(then contrasts use the seeds present in both arms and report n).
Usage: python scripts/analyze-amendment5.py [--logs DIR] [--out DIR] [--allow-incomplete]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [7, 42, 123, 777, 888, 999, 2026, 5150, 31415, 60486]
LM_CELLS = {"B24457k": 4, "B42487k": 8}
VIS_CELLS = {"B7333k": 4, "B17350k": 8}
LM_EP = {"final": "final_test_ppl", "bestval": "test_ppl_at_best_val"}
VIS_EP = {"final": "final_test_acc", "bestval": "test_acc_at_best_val"}
VIS_MARGIN = 0.005
LM_MARGIN_FRAC = 0.01
OUTPUTS = ("a5_runs.csv", "a5_tests.csv", "a5_buildcheck.csv", "a5_diagnostics.csv")


# ── statistics (paired_p, holm, tost_paired, gap_cv: as in reproduce_numbers_v11.py) ──
def paired_p(a, b):
    """Two-sided paired t-test p; identical vectors -> 1.0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0) or d.std(ddof=1) == 0:
        return 1.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def holm(p):
    p = np.asarray(p, float)
    adj, cur = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p, kind="mergesort")):
        cur = max(cur, (len(p) - k) * p[i])
        adj[i] = min(1.0, cur)
    return adj


def tost_paired(d, margin):
    """Two one-sided paired t-tests of |mean diff| < margin; p_tost = max of the two; 90% CI."""
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    p_lower = 1 - stats.t.cdf((m + margin) / se, df)  # H0: diff <= -margin
    p_upper = stats.t.cdf((m - margin) / se, df)      # H0: diff >= +margin
    h = stats.t.ppf(0.95, df) * se
    return float(max(p_lower, p_upper)), float(m - h), float(m + h), float(p_lower), float(p_upper)


def sign_p(d):
    """Two-sided exact sign test on the nonzero paired differences (zeros dropped)."""
    d = np.asarray(d, float)
    pos, n = int((d > 0).sum()), int((d != 0).sum())
    return 1.0 if n == 0 else float(stats.binomtest(pos, n, 0.5).pvalue)


def gap_cv(routes):
    """Mean over (layer, expert with >2 activations) of std(gaps, ddof=0)/mean(gaps)."""
    cvs = []
    for l in range(routes.shape[1]):
        for e in range(8):
            idx = np.where(routes[:, l] == e)[0]
            if len(idx) > 2:
                g = np.diff(idx)
                cvs.append(g.std() / g.mean())
    return float(np.mean(cvs))


# ── run inventory ─────────────────────────────────────────────────────────────
def expected_runs():
    """(name, role, family, arm, cell, seed, endpoint keys)."""
    rows = []
    for c in LM_CELLS:
        for s in SEEDS:
            rows.append((f"e6lm5_gated_top1_{c}_s{s}", "a5", "lm", "gated_top1", c, s, LM_EP))
            for m in ("round_robin", "learned_top1"):
                rows.append((f"e6lm3_{m}_{c}_s{s}", "comparator", "lm", m, c, s, LM_EP))
    for c in VIS_CELLS:
        for s in SEEDS:
            rows.append((f"e1a5_c100_gated_top1_{c}_s{s}", "a5", "vision", "gated_top1", c, s, VIS_EP))
            for m in ("round_robin", "learned_top1"):
                rows.append((f"e1_c100_{m}_{c}_s{s}", "comparator", "vision", m, c, s, VIS_EP))
    for s in SEEDS:
        rows.append((f"e1a5_c100_dense_B17350k_s{s}", "a5", "dense", "dense", "B17350k", s, VIS_EP))
        rows.append((f"e1_c100_round_robin_B3907k_s{s}", "comparator", "dense", "round_robin",
                     "B3907k", s, VIS_EP))
    for m in ("round_robin", "learned_top1"):
        rows.append((f"e1a5chk_c100_{m}_B17350k_s42", "buildcheck", "vision", m, "B17350k", 42, VIS_EP))
    return rows


def load(logs, name, ep):
    p = logs / name / "results.json"
    if not p.exists():
        return None, f"{name}: results.json missing"
    r = json.load(open(p))
    miss = [k for k in list(ep.values()) + ["best_val_epoch", "epochs"] if k not in r]
    if miss:
        return None, f"{name}: keys missing {miss}"
    return r, None


def budget_of(r):
    c = r.get("config", {})
    return c.get("budget_bytes", c.get("budget"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(ROOT / "result" / "logs"))
    ap.add_argument("--out", default=str(ROOT / "result" / "tables" / "a5"))
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    logs, out = Path(a.logs), Path(a.out)
    clash = [f for f in OUTPUTS if (out / f).exists()]
    if clash:
        sys.exit(f"refusing to overwrite {out}: {clash}")

    runs, problems = {}, []
    for name, role, fam, arm, cell, seed, ep in expected_runs():
        r, err = load(logs, name, ep)
        if err:
            problems.append(err)
            continue
        if role == "a5" and fam == "lm" and not (logs / name / "routes_train.npy").exists():
            problems.append(f"{name}: routes_train.npy missing")
        runs[name] = dict(name=name, role=role, family=fam, arm=arm, cell=cell, seed=seed, r=r,
                          final=float(r[ep["final"]]), bestval=float(r[ep["bestval"]]))
    n_exp = len(expected_runs())
    print(f"expected runs: {n_exp}; loaded: {len(runs)}; problems: {len(problems)}")
    for p in problems:
        print("  MISSING", p)
    if problems and not a.allow_incomplete:
        sys.exit(1)

    # build check first: it decides whether the e1_ comparators are the declared ones
    bc_rows, bc_ok = [], True
    for m in ("round_robin", "learned_top1"):
        new, old = f"e1a5chk_c100_{m}_B17350k_s42", f"e1_c100_{m}_B17350k_s42"
        A, O = runs.get(new), runs.get(old)
        row = {"rerun": new, "original": old}
        for k in ("final_test_acc", "test_acc_at_best_val"):
            va = A["r"][k] if A else ""
            vo = O["r"][k] if O else ""
            eq = (A is not None and O is not None and va == vo)
            row.update({f"{k}_rerun": va, f"{k}_original": vo, f"{k}_equal": "yes" if eq else "no"})
            bc_ok &= eq
        bc_rows.append(row)
    vis_valid = "yes" if bc_ok else "no"
    print(f"build check exact: {bc_ok}")

    def cellvals(fam, arm, cell, key):
        return {v["seed"]: v[key] for v in runs.values()
                if v["family"] == fam and v["arm"] == arm and v["cell"] == cell and v["role"] != "buildcheck"}

    tests = []

    def family(fam_label, pairs, margin_fn, valid):
        for key in ("final", "bestval"):
            rows = []
            for (fa, aa, ca), (fb, ab, cb) in pairs:
                A, B = cellvals(fa, aa, ca, key), cellvals(fb, ab, cb, key)
                s = sorted(set(A) & set(B))
                if len(s) < 2:
                    print(f"  SKIP {fam_label} {aa}@{ca} vs {ab}@{cb} ({key}): n={len(s)}")
                    continue
                x, y = np.array([A[i] for i in s]), np.array([B[i] for i in s])
                d = x - y
                row = {"family": fam_label, "endpoint": key, "arm": aa, "arm_cell": ca,
                       "comparator": ab, "comparator_cell": cb, "n": len(s),
                       "arm_mean": x.mean(), "comparator_mean": y.mean(), "mean_diff": d.mean(),
                       "sd_diff": d.std(ddof=1), "seeds_positive": int((d > 0).sum()),
                       "seeds_zero": int((d == 0).sum()), "p_t": paired_p(x, y), "p_sign": sign_p(d),
                       "comparator_valid": valid}
                m = margin_fn(ab, y)
                if m is not None and d.std(ddof=1) > 0:
                    pt, lo, hi, pl, pu = tost_paired(d, m)
                    row.update({"tost_margin": m, "p_tost": pt, "ci90_lo": lo, "ci90_hi": hi,
                                "p_tost_lower": pl, "p_tost_upper": pu})
                rows.append(row)
            for row, padj in zip(rows, holm([r["p_t"] for r in rows]) if rows else []):
                row["p_holm"] = float(padj)
                row["holm_family_size"] = len(rows)
            tests.extend(rows)

    lm_pairs = [(("lm", "gated_top1", c), ("lm", m, c)) for c in LM_CELLS for m in ("round_robin", "learned_top1")]
    family("lm", lm_pairs, lambda ab, y: LM_MARGIN_FRAC * y.mean() if ab == "round_robin" else None, "yes")
    vis_pairs = [(("vision", "gated_top1", c), ("vision", m, c)) for c in VIS_CELLS
                 for m in ("round_robin", "learned_top1")]
    family("vision", vis_pairs, lambda ab, y: VIS_MARGIN if ab == "round_robin" else None, vis_valid)
    dense_pairs = [(("dense", "dense", "B17350k"), ("dense", "round_robin", "B3907k")),
                   (("dense", "dense", "B17350k"), ("vision", "round_robin", "B17350k"))]
    family("dense", dense_pairs, lambda ab, y: None, vis_valid)

    # per-run table
    run_rows = []
    for v in sorted(runs.values(), key=lambda v: (v["role"], v["family"], v["arm"], v["cell"], v["seed"])):
        r = v["r"]
        peak = max(e["peak_sram_bytes"] for e in r["epochs"])
        b = budget_of(r)
        run_rows.append({"name": v["name"], "role": v["role"], "family": v["family"], "arm": v["arm"],
                         "cell": v["cell"], "seed": v["seed"], "final_metric": v["final"],
                         "bestval_metric": v["bestval"], "best_val_epoch": r["best_val_epoch"] + 1,
                         "n_epochs": len(r["epochs"]), "peak_sram_bytes": peak, "budget_bytes": b,
                         "peak_within_budget": "yes" if (b is not None and peak <= b) else "no"})

    # diagnostics: LM gated arm
    diag = []
    for c in LM_CELLS:
        sel = [v for v in runs.values() if v["family"] == "lm" and v["arm"] == "gated_top1" and v["cell"] == c]
        for v in sorted(sel, key=lambda v: v["seed"]):
            r = v["r"]
            rp = logs / v["name"] / "routes_train.npy"
            cv = gap_cv(np.load(rp)) if rp.exists() else float("nan")
            ds = r["epochs"][-1].get("debt_stats")
            ent = (min(d["utilization_entropy"] for d in ds)
                   if ds and all("utilization_entropy" in d for d in ds) else float("nan"))
            diag.append({"cell": c, "seed": v["seed"], "best_val_epoch": r["best_val_epoch"] + 1,
                         "n_epochs": len(r["epochs"]),
                         "best_is_last": "yes" if r["best_val_epoch"] + 1 == len(r["epochs"]) else "no",
                         "gap_cv": cv, "util_entropy_min_layers": ent})
        if sel:
            dd = [d for d in diag if d["cell"] == c]
            diag.append({"cell": c, "seed": "ALL", "best_val_epoch": "",
                         "n_epochs": "", "best_is_last": sum(d["best_is_last"] == "yes" for d in dd),
                         "gap_cv": float(np.mean([d["gap_cv"] for d in dd])),
                         "util_entropy_min_layers": float(np.min([d["util_entropy_min_layers"] for d in dd]))})

    out.mkdir(parents=True, exist_ok=True)

    def write(fname, rows):
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(out / fname, "x", newline="") as f:  # "x": never overwrite
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{x:.10g}" if isinstance(x, float) else x) for k, x in r.items()})
        print(f"wrote {out / fname} ({len(rows)} rows)")

    write("a5_runs.csv", run_rows)
    write("a5_tests.csv", tests)
    write("a5_buildcheck.csv", bc_rows)
    write("a5_diagnostics.csv", diag)
    if problems:
        print(f"INCOMPLETE: {len(problems)} expected runs missing (--allow-incomplete)")


if __name__ == "__main__":
    main()
