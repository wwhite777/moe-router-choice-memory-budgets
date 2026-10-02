"""Count the LM's BPTT windows per epoch (training, validation, test) — Appendix D arithmetic.

The superseded fixed-cadence router advanced one position per evaluation window (pre-Amendment-4
evaluation), so its per-epoch phase drift was (training + validation + test windows) mod |F|,
against (training windows) mod |F| for round-robin and the corrected router. Windows are counted
exactly as lm_train_loop.py iterates them: full windows only, ((n_steps - 1) // bptt) per split
after batchify(ids, batch_size).

Usage: python scripts/count-lm-windows-v11.py --data-dir ./data --out result/tables/lm_v11/lm_windows.csv
Needs the WikiText-2 data (downloaded by the package on first use). Refuses to overwrite.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatedmoe.lm_model_define import load_wikitext2, batchify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--out", required=True)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--bptt", type=int, default=128)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"refusing to overwrite existing {out}")

train, valid, test, vocab = load_wikitext2(a.data_dir)
rows, win = [], {}
for name, ids in (("train", train), ("valid", valid), ("test", test)):
    b = batchify(ids, a.batch_size)
    n = (b.shape[1] - 1) // a.bptt
    win[name] = n
    rows.append({"quantity": f"windows_{name}", "value": n,
                 "definition": f"{ids.numel()} tokens -> batchify({a.batch_size}) -> {b.shape[1]} steps per stream; "
                               f"full windows of {a.bptt}"})
ev = win["valid"] + win["test"]
tot = win["train"] + ev
rows += [
    {"quantity": "windows_eval_per_epoch", "value": ev, "definition": "validation + test windows (both evaluated every epoch)"},
    {"quantity": "windows_all_per_epoch", "value": tot, "definition": "training + evaluation windows"},
]
for F in (4, 8):
    rows += [
        {"quantity": f"drift_train_mod{F}", "value": win["train"] % F,
         "definition": f"per-epoch phase drift of a cycle over {F} experts that advances only in training"},
        {"quantity": f"drift_all_mod{F}", "value": tot % F,
         "definition": f"per-epoch drift when evaluation also advances the cycle (pre-Amendment-4 fixed-cadence)"},
    ]
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "x", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["quantity", "value", "definition"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
