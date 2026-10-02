"""Regenerate both LM figure panels from the CSVs written by scripts/analyze-lm-v11.py.

Runs, one after the other (stops at the first failure):
  plot_fig1_threshold_v11.py --summary S --out-dir D   (Fig 1; panel b = LM)
  plot_fig2_cadence_v11.py   --summary S --curves C --out-dir D   (Fig 2, both panels)

Usage: python3 scripts/plot_fig_lm_v11.py --summary <lm_summary.csv> --curves <lm_curves.csv> --out-dir <dir>
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--curves", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
jobs = [[sys.executable, str(HERE / "plot_fig1_threshold_v11.py"), "--summary", a.summary, "--out-dir", a.out_dir],
        [sys.executable, str(HERE / "plot_fig2_cadence_v11.py"), "--summary", a.summary, "--curves", a.curves,
         "--out-dir", a.out_dir]]
for cmd in jobs:
    print("RUN", " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd)), flush=True)
    rc = subprocess.call(cmd)
    print(f"EXIT {rc}", flush=True)
    if rc:
        sys.exit(rc)
