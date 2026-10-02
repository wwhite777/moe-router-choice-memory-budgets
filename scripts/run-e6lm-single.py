"""Run a single E6-LM experiment (WikiText-2 MoE-transformer)."""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import LOG_DIR
from gatedmoe.lm_train_loop import run_lm_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bptt", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff-shared", type=int, default=256)
    ap.add_argument("--widths", type=str,
                    default="64,96,128,192,256,384,512,1024")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    cfg = dict(
        method=args.method, budget=args.budget, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, bptt=args.bptt,
        d_model=args.d_model, n_heads=args.n_heads,
        d_ff_shared=args.d_ff_shared,
        widths=[int(w) for w in args.widths.split(",")],
        num_layers=args.num_layers, lr=args.lr, name=args.name,
        data_dir="./data", save_dir=str(LOG_DIR),
    )
    print(f"Running LM: {cfg['name']} | {cfg['method']} | "
          f"budget={cfg['budget'] // 1024}KB | seed={cfg['seed']}", flush=True)
    r = run_lm_experiment(cfg)
    print(f"Final test ppl: {r['final_test_ppl']:.2f}")


if __name__ == "__main__":
    main()
