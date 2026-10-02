"""Boundary-seed top-ups (PROTOCOL.md §2): +5 seeds at transition and
all-feasible cells so the decisive cells reach n=10 before gate G-SM1.

Cells: S1 m in {6,8}; e1 |F|=8 (both datasets); e4 |F| in {3,6,8}
(budgets 320K/512K/2M). Jobs are cloned from the frozen base grids with the
confirmatory seed block; names carry the new seed so is_done() resumption
works unchanged.
"""

import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import CONFIG_DIR

TOPUP_SEEDS = [60486, 5150, 777, 888, 999]

SELECT = {
    "s1": lambda j: j["expected_fsize"] in (6, 8),
    "e1": lambda j: j["expected_fsize"] == 8,
    "e4": lambda j: j["budget"] in (320 * 1024, 512 * 1024, 2048 * 1024),
}


def main():
    jobs = []
    for grid, keep in SELECT.items():
        base = json.load(open(CONFIG_DIR / f"{grid}_grid.json"))["jobs"]
        templates = [j for j in base if j["seed"] == 42 and keep(j)]
        for t in templates:
            for rank, seed in enumerate(TOPUP_SEEDS):
                j = dict(t)
                j["seed"] = seed
                j["seed_rank"] = rank
                j["name"] = re.sub(r"_s42$", f"_s{seed}", t["name"])
                jobs.append(j)
        print(f"{grid}: {len(templates)} template cells -> {len(templates) * len(TOPUP_SEEDS)} runs")
    with open(CONFIG_DIR / "topup_grid.json", "w") as f:
        json.dump({"jobs": jobs}, f, indent=1)
    print(f"topup total: {len(jobs)} runs")


if __name__ == "__main__":
    main()
