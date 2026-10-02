"""Resumable queue runner for the E1/E4 grids over multiple GPUs.

Reads configs/e1_grid.json + configs/e4_grid.json, skips completed runs
(results.json with final_test_acc), and keeps SLOTS_PER_GPU subprocesses per
GPU busy. Seed-rank 0 jobs run first so the full gap-vs-|F| curve appears
early. Each run writes its own train.log next to its results.json.

Usage: python3 scripts/run-e1-queue.py [--gpus 0,2,3] [--slots 3] [--grids e1,e4]
"""

import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR, CONFIG_DIR, PROJECT_ROOT


def is_done(name: str) -> bool:
    p = Path(LOG_DIR) / name / "results.json"
    if not p.exists():
        return False
    try:
        j = json.load(open(p))
        return "final_test_acc" in j or "final_test_ppl" in j
    except Exception:
        return False


def job_cmd(job: dict) -> list:
    if job["name"].startswith("e6lm"):
        cmd = [
            sys.executable, "scripts/run-e6lm-single.py",
            "--method", job["method"],
            "--budget", str(job["budget"]),
            "--seed", str(job["seed"]),
            "--name", job["name"],
        ]
        if job.get("d_model"):
            cmd += ["--d-model", str(job["d_model"])]
        if job.get("d_ff_shared"):
            cmd += ["--d-ff-shared", str(job["d_ff_shared"])]
        return cmd
    cmd = [
        sys.executable, "scripts/run-train-single.py",
        "--method", job["method"],
        "--dataset", job["dataset"],
        "--budget", str(job["budget"]),
        "--seed", str(job["seed"]),
        "--epochs", str(job["epochs"]),
        "--batch-size", str(job["batch_size"]),
        "--d-model", str(job["d_model"]),
        "--d-ff-shared", str(job["d_ff_shared"]),
        "--patch-size", str(job["patch_size"]),
        "--widths", ",".join(str(w) for w in job["widths"]),
        "--num-workers", "2",
        "--name", job["name"],
    ]
    if job.get("admissible") is not None:
        cmd += ["--admissible", ",".join(str(i) for i in job["admissible"])]
    if job.get("capacity_factor") is not None:
        cmd += ["--capacity-factor", str(job["capacity_factor"])]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    # Default GPU ids; override with --gpus.
    ap.add_argument("--gpus", type=str, default="0,2")
    ap.add_argument("--slots", type=int, default=3, help="concurrent runs per GPU")
    ap.add_argument("--grids", type=str, default="s1,e1,e4")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    jobs = []
    grid_order = [g.strip() for g in args.grids.split(",")]
    for prio, grid in enumerate(grid_order):
        grid_path = CONFIG_DIR / f"{grid}_grid.json"
        for j in json.load(open(grid_path))["jobs"]:
            j["grid_prio"] = prio
            jobs.append(j)

    # All grids at seed #1 before any seed #2; S1 before S2/E4 within a rank.
    jobs.sort(key=lambda j: (j["seed_rank"], j["grid_prio"],
                             j.get("dataset", ""), j["budget"], j["name"]))
    todo = [j for j in jobs if not is_done(j["name"])]
    print(f"{len(jobs)} jobs in grids; {len(jobs) - len(todo)} already done; "
          f"{len(todo)} to run on GPUs {gpus} x {args.slots} slots", flush=True)

    slots = []  # list of (proc, job, gpu, t0)
    launched = 0
    failed = []

    def launch(job, gpu):
        run_dir = Path(LOG_DIR) / job["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_f = open(run_dir / "train.log", "w")
        p = subprocess.Popen(job_cmd(job), env=env, stdout=log_f,
                             stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        return p

    idx = 0
    while idx < len(todo) or slots:
        # Reap finished
        for entry in slots[:]:
            proc, job, gpu, t0 = entry
            if proc.poll() is not None:
                slots.remove(entry)
                dt = (time.time() - t0) / 60
                if proc.returncode == 0 and is_done(job["name"]):
                    print(f"DONE  {job['name']} ({dt:.1f} min)", flush=True)
                else:
                    failed.append(job["name"])
                    print(f"FAIL  {job['name']} rc={proc.returncode} ({dt:.1f} min)", flush=True)
        # Fill free slots
        while idx < len(todo) and len(slots) < len(gpus) * args.slots:
            per_gpu = {g: sum(1 for _, _, gg, _ in slots if gg == g) for g in gpus}
            gpu = min(gpus, key=lambda g: per_gpu[g])
            job = todo[idx]
            idx += 1
            print(f"START {job['name']} on GPU {gpu} [{idx}/{len(todo)}]", flush=True)
            slots.append((launch(job, gpu), job, gpu, time.time()))
        time.sleep(10)

    print(f"Queue complete. {len(todo) - len(failed)} ok, {len(failed)} failed.", flush=True)
    for name in failed:
        print(f"  FAILED: {name}", flush=True)


if __name__ == "__main__":
    main()
