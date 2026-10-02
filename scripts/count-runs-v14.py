"""Run inventory over every run prefix, including the Amendment 5 runs (e6lm5, e1a5, e1a5chk).

The inventory rows of reproduce_numbers_v11.py cover only the runs before Amendment 5;
this script counts all of them and scans every run for completeness, finite endpoints and
budget compliance. Loading, the infrastructure-failure definition, the training-failure scan
and the analytic peak are those of reproduce_numbers_v11.py.

Reads <logs>/<run>/results.json (and routes_train.npy or route_class_counts.npy for the experts
a budgeted vision run used) and configs/a5_grid.json. Writes to <out-dir> (default
result/tables/v14/; refuses to overwrite):
  inventory_v14.csv          one row per prefix: role (paper | superseded), level (batch | token),
                             run directories, results.json files, runs with all configured epochs,
                             runs with a finite final and a finite best-validation test metric, and
                             for budgeted batch-level vision runs (dense excluded) the number with
                             peak <= budget and with peak == analytic sum (empty for other prefixes)
  inventory_v14_summary.csv  quantity, value, definition (totals, Amendment 5 grid completeness,
                             failures, budget compliance, dense ceiling peak vs budget)
Checks (exit 1, nothing written): every directory has results.json; every paper run is complete
with finite endpoints; every Amendment 5 job is complete; every budgeted vision run has
peak <= budget.
Usage: python3 scripts/count-runs-v14.py [--logs DIR] [--grid FILE] [--out-dir DIR]
Dependencies: numpy (standard library otherwise).
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

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "result" / "logs"
GRID = ROOT / "configs" / "a5_grid.json"
OUT_DEFAULT = ROOT / "result" / "tables" / "v14"
OUTPUTS = ("inventory_v14.csv", "inventory_v14_summary.csv")

PREFIX_ORDER = ["e6lm3d256", "e6lmd256", "e1a5chk", "e6lm3", "e6lm2", "e6lm5", "e5tune", "e1a5",
                "e6lm", "e5", "e4", "e1", "s1"]  # longest first
LM_ARMS = ("e6lm", "e6lm2", "e6lmd256", "e6lm3", "e6lm3d256", "e6lm5")
SUPERSEDED = ("e6lm2", "e6lmd256")
TOKEN_LEVEL = ("e5", "e5tune")
ORIGINAL_BUDGETED = ("s1", "e1", "e4")
A5_BUDGETED = ("e1a5", "e1a5chk")  # dense runs excluded below
VIS_PAT = re.compile(r"^(s1|e1|e4|e5|e1a5|e1a5chk)_(c10|c100)_(\w+?)_(m\d+v\d+|B\d+k)_s(\d+)$")
LM_PAT = re.compile(r"^(e6lm3d256|e6lmd256|e6lm3|e6lm2|e6lm5|e6lm)_(\w+?)_B(\d+)k_s(\d+)$")
PATCH = {"s1": 4, "e1": 4, "e4": 8, "e5": 4, "e1a5": 4, "e1a5chk": 4}  # S = (32/p)^2 tokens
TAU = 4  # bytes per fp32 value
DENSE = "dense"


def chat(b, S, d, f, tau=TAU):
    """Closed-form training cost of a Linear-ReLU-Linear block: 2P + A (1-byte ReLU mask)."""
    P = tau * (2 * d * f + f + d)
    A = tau * b * S * (2 * d + 2 * f) + b * S * f
    return 2 * P + A


def prefix_of(name):
    return next((p for p in PREFIX_ORDER if name.startswith(p + "_")), None)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_runs(logs):
    runs = []
    for d in sorted(logs.iterdir()):
        p = prefix_of(d.name)
        if p is None or not d.is_dir():
            continue
        r = {"name": d.name, "prefix": p, "dir": d, "ok": (d / "results.json").exists()}
        if r["ok"]:
            j = json.load(open(d / "results.json"))
            cfg, eps = j.get("config", {}), j.get("epochs", [])
            r.update(cfg=cfg, n_ep=len(eps), peaks=[e.get("peak_sram_bytes") for e in eps])
            if p in LM_ARMS:
                m = LM_PAT.match(d.name)
                if m is None:
                    sys.exit(f"run name does not parse: {d.name}")
                g, meth, bkb, seed = m.groups()
                r.update(kind="lm", method=meth, seed=int(seed),
                         final=j.get("final_test_ppl"), at_best_val=j.get("test_ppl_at_best_val"))
            else:
                if p == "e5tune":
                    ds, cf, s = d.name.split("_")[1:4]
                    r.update(dataset=ds, method="token_capacity", cell=cf, seed=int(s[1:]))
                else:
                    m = VIS_PAT.match(d.name)
                    if m is None or m.group(1) != p:
                        sys.exit(f"run name does not parse: {d.name}")
                    g, ds, meth, cell, seed = m.groups()
                    r.update(dataset=ds, method=meth, cell=cell, seed=int(seed))
                r.update(kind="vision", final=j.get("final_test_acc"), at_best_val=j.get("test_acc_at_best_val"))
        runs.append(r)
    return runs


def complete(r):
    return r["ok"] and r["n_ep"] == r["cfg"].get("epochs")


def infra_failures(runs):
    """As in reproduce_numbers_v11.py: no results.json, fewer epochs than configured, or a non-finite final."""
    return sum(1 for r in runs if not r["ok"]) + sum(
        1 for r in runs if r["ok"] and (r["n_ep"] != r["cfg"].get("epochs") or r["final"] is None
                                         or not math.isfinite(r["final"])))


def failure_scan(ok_runs):
    """As in reproduce_numbers_v11.py: compare each run with its sibling seeds (same config)."""
    groups = defaultdict(list)
    for r in ok_runs:
        groups[r["name"].rsplit("_s", 1)[0]].append(r)
    hits = []
    for key, rs in groups.items():
        for r in rs:
            sib = [x["final"] for x in rs if x is not r]
            if not sib:
                continue
            med = float(np.median(sib))
            if (r["kind"] == "vision" and r["final"] < med - 0.10) or (
                    r["kind"] == "lm" and (not math.isfinite(r["final"]) or r["final"] > 2 * med)):
                hits.append((r, sib))
    return hits


def used_experts(r):
    """Experts in the training route trace (routes_train.npy; -1 = null path), else route_class_counts.npy."""
    tr = r["dir"] / "routes_train.npy"
    if tr.exists():
        return [int(x) for x in np.unique(np.load(tr)) if x >= 0]
    rcc = np.load(r["dir"] / "route_class_counts.npy")  # [L, K+1, C], slot 0 = null path
    return [e for e in range(rcc.shape[1] - 1) if rcc[:, e + 1].sum() > 0]


def budget_check(r):
    """(peak <= budget, peak == analytic) for a budgeted batch-level vision run (definition of reproduce_numbers_v11.py)."""
    c = r["cfg"]
    S = (32 // PATCH[r["prefix"]]) ** 2
    widths = c["d_ff_expert"] if isinstance(c["d_ff_expert"], list) else [c["d_ff_expert"]] * c["num_experts"]
    used = used_experts(r)
    analytic = chat(c["batch_size"], S, c["d_model"], c["d_ff_shared"]) + \
        (max(chat(c["batch_size"], S, c["d_model"], widths[u]) for u in used) if used else 0)
    peak = max(r["peaks"])
    return peak <= c["budget_bytes"], peak == analytic


def budgeted(r):
    return r["ok"] and r["prefix"] in ORIGINAL_BUDGETED + A5_BUDGETED and r["method"] != DENSE


def rel(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(LOGS))
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    logs, out = Path(a.logs), Path(a.out_dir)
    clash = [f for f in OUTPUTS if (out / f).exists()]
    if clash:
        sys.exit(f"refusing to overwrite {rel(out)}: {clash}")

    runs = load_runs(logs)
    by = defaultdict(list)
    for r in runs:
        by[r["prefix"]].append(r)
    problems = []

    inv, bud = [], {}
    for p in sorted(by):
        rs = by[p]
        ok = [r for r in rs if r["ok"]]
        row = {"prefix": p, "role": "superseded" if p in SUPERSEDED else "paper",
               "level": "token" if p in TOKEN_LEVEL else "batch", "n_dirs": len(rs),
               "n_results_json": len(ok), "n_complete_epochs": sum(complete(r) for r in rs),
               "n_finite_final": sum(finite(r["final"]) for r in ok),
               "n_finite_bestval": sum(finite(r["at_best_val"]) for r in ok)}
        bv = [r for r in ok if budgeted(r)]
        if p in ORIGINAL_BUDGETED + A5_BUDGETED:
            chk = [budget_check(r) for r in bv]
            bud[p] = (len(bv), sum(c[0] for c in chk), sum(c[1] for c in chk))
            for r, c in zip(bv, chk):
                if not c[0]:
                    problems.append(f"{r['name']}: peak above budget")
            row.update(n_budgeted_vision=bud[p][0], n_peak_le_budget=bud[p][1], n_peak_eq_analytic=bud[p][2])
        else:
            row.update(n_budgeted_vision="", n_peak_le_budget="", n_peak_eq_analytic="")
        inv.append(row)
        if len(ok) != len(rs):
            problems.append(f"{p}: {len(rs) - len(ok)} run directories without results.json")

    paper = [r for r in runs if r["prefix"] not in SUPERSEDED]
    n_inf = infra_failures(paper)
    if n_inf:
        problems.append(f"{n_inf} infrastructure failures among the paper runs")
    nonfinite_bv = [r["name"] for r in paper if r["ok"] and not finite(r["at_best_val"])]
    hits = failure_scan([r for r in paper if r["ok"]])

    jobs = json.load(open(a.grid))["jobs"]
    names = {r["name"]: r for r in runs}
    done = [j["name"] for j in jobs if j["name"] in names and complete(names[j["name"]])
            and finite(names[j["name"]]["final"]) and finite(names[j["name"]]["at_best_val"])]
    if len(done) != len(jobs):
        problems.append(f"Amendment 5 grid: {len(done)}/{len(jobs)} jobs complete; missing or incomplete: "
                        + ";".join(sorted({j['name'] for j in jobs} - set(done))))

    dense = [r for r in runs if r["ok"] and r["prefix"] == "e1a5" and r["method"] == DENSE]
    dense_budgets = sorted({r["cfg"]["budget_bytes"] for r in dense})
    if len(dense_budgets) != 1:
        problems.append(f"dense runs: expected one configured budget, found {dense_budgets}")

    print("run directories by prefix: " + ", ".join(f"{r['prefix']} {r['n_dirs']}" for r in inv))
    if problems:
        for p in problems:
            print("CHECK FAILED:", p)
        sys.exit(1)

    cnt = {r["prefix"]: r["n_dirs"] for r in inv}
    n_all = sum(cnt.values())
    n_paper = sum(v for k, v in cnt.items() if k not in SUPERSEDED)
    n_batch = n_paper - sum(cnt.get(k, 0) for k in TOKEN_LEVEL)
    o = [bud[p] for p in ORIGINAL_BUDGETED]
    a5 = [bud[p] for p in A5_BUDGETED]
    dpeak = max(max(r["peaks"]) for r in dense)
    dbud = dense_budgets[0]
    summ = [
        ("total_all", n_all, "run directories over all prefixes (" + ", ".join(sorted(cnt)) + ")"),
        ("total_paper", n_paper, "total_all minus the superseded prefixes " + ", ".join(SUPERSEDED)),
        ("total_batch_level", n_batch, "total_paper minus the token-level prefixes " + ", ".join(TOKEN_LEVEL)),
        ("a5_jobs_in_grid", len(jobs), "jobs in configs/a5_grid.json"),
        ("a5_jobs_complete", len(done), "grid jobs with results.json, all configured epochs, finite final and best-validation test metric"),
        ("infra_failures_paper", n_inf, "paper runs without results.json, with fewer epochs than configured, or with a non-finite final endpoint"),
        ("nonfinite_bestval_paper", len(nonfinite_bv), "paper runs without a finite test metric at the best-validation epoch (reported, not a failure)"),
        ("failure_scan_hits_paper", len(hits), "vision: final test acc < median of sibling seeds (same config) - 0.10; LM: non-finite or > 2x sibling median ppl"),
        ("failure_scan_hit_names_paper", ";".join(sorted(h[0]["name"] for h in hits)), "runs hit by the scan above"),
        ("failure_scan_hit_finals_paper", ";".join(repr(float(h[0]["final"])) for h in sorted(hits, key=lambda h: h[0]["name"])),
         "final endpoint of each run hit, same order"),
        ("budgeted_vision_original", sum(x[0] for x in o), "budgeted batch-level vision runs, prefixes " + ", ".join(ORIGINAL_BUDGETED)),
        ("budgeted_vision_original_peak_le_budget", sum(x[1] for x in o), "max over epochs of peak_sram_bytes <= config budget_bytes"),
        ("budgeted_vision_original_peak_eq_analytic", sum(x[2] for x in o),
         "peak == C(shared) + max C(expert) over experts used in training, C = 2P + A at the training batch size"),
        ("budgeted_vision_a5", sum(x[0] for x in a5), "budgeted batch-level vision runs, prefixes " + ", ".join(A5_BUDGETED) + " (dense runs excluded)"),
        ("budgeted_vision_a5_peak_le_budget", sum(x[1] for x in a5), "same definition"),
        ("budgeted_vision_a5_peak_eq_analytic", sum(x[2] for x in a5), "same definition"),
        ("dense_runs", len(dense), "e1a5 runs with method dense (capacity ceiling; ignores the memory gate)"),
        ("dense_peak_bytes_max", dpeak, "max over dense runs and epochs of peak_sram_bytes"),
        ("dense_budget_bytes", dbud, "config budget_bytes of the dense runs (the |F|=8 configuration)"),
        ("dense_peak_over_budget", repr(dpeak / dbud), "dense_peak_bytes_max / dense_budget_bytes"),
    ]
    summ = sorted(({"quantity": q, "value": v, "definition": d} for q, v, d in summ), key=lambda x: x["quantity"])

    out.mkdir(parents=True, exist_ok=True)
    with open(out / OUTPUTS[0], "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)
    with open(out / OUTPUTS[1], "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "definition"])
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(f"{s['quantity']} = {s['value']}")
    print(f"wrote {rel(out / OUTPUTS[0])} ({len(inv)} rows), {rel(out / OUTPUTS[1])} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
