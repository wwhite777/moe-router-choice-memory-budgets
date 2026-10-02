"""G-SM1 gate top-ups (decisions D-A and D-B, gate doc 2026-08-22).

D-A: vision full-grid n=10 — confirmatory seeds {60486,5150,777,888,999} for
     every s1/e1/e4 cell not already topped up.
D-B: E4 boundary cells (320K/512K/2048K) to n=20 with a NEW pre-declared seed
     block {1111,...,10101} (declared here, before any of these runs exist).
"""

import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import CONFIG_DIR

CONF_SEEDS = [60486, 5150, 777, 888, 999]
E4_N20_SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10101]
ALREADY = json.load(open(CONFIG_DIR / "topup_grid.json"))["jobs"]
ALREADY_NAMES = {j["name"] for j in ALREADY}


def clone(t, seed, rank):
    j = dict(t)
    j["seed"] = seed
    j["seed_rank"] = rank
    j["name"] = re.sub(r"_s42$", f"_s{seed}", t["name"])
    return j


def main():
    jobs = []
    for grid in ("s1", "e1", "e4"):
        base = json.load(open(CONFIG_DIR / f"{grid}_grid.json"))["jobs"]
        templates = [j for j in base if j["seed"] == 42]
        for t in templates:
            for rank, seed in enumerate(CONF_SEEDS):
                j = clone(t, seed, rank)
                if j["name"] not in ALREADY_NAMES:
                    jobs.append(j)
    print(f"D-A vision full-grid n=10: {len(jobs)} new runs")

    e4 = json.load(open(CONFIG_DIR / "e4_grid.json"))["jobs"]
    e4_boundary = [j for j in e4 if j["seed"] == 42
                   and j["budget"] in (320 * 1024, 512 * 1024, 2048 * 1024)]
    n20 = []
    for t in e4_boundary:
        for rank, seed in enumerate(E4_N20_SEEDS):
            n20.append(clone(t, seed, rank + 5))  # after D-A ranks
    print(f"D-B e4 boundary n=20: {len(n20)} runs")
    jobs += n20

    with open(CONFIG_DIR / "gate_topup_grid.json", "w") as f:
        json.dump({"jobs": jobs}, f, indent=1)
    print(f"gate topup total: {len(jobs)}")


if __name__ == "__main__":
    main()
