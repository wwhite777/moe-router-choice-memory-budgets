"""Run a single training experiment from a YAML config or command-line args."""

import sys
import argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from gatedmoe.util_config_io import ExperimentConfig, load_config, save_config, LOG_DIR
from gatedmoe.core_train_loop import run_experiment
from gatedmoe.core_model_define import build_model_from_config
from gatedmoe.sim_baseline_routers import create_baseline_model


def main():
    parser = argparse.ArgumentParser(description="GatedMoE single experiment")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--budget", type=int, default=None, help="SRAM budget in bytes")
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--budget-schedule", type=str, default=None,
                        help="Budget schedule: static or alternate")
    parser.add_argument("--budget-min", type=int, default=None,
                        help="Minimum SRAM budget in bytes for dynamic schedules")
    parser.add_argument("--budget-cycle", type=int, default=None,
                        help="Number of steps per budget block")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff-shared", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--widths", type=str, default=None,
                        help="Comma-separated expert hidden widths (sets num_experts)")
    parser.add_argument("--admissible", type=str, default=None,
                        help="Comma-separated admissible expert ids (S1 masks)")
    parser.add_argument("--capacity-factor", type=float, default=None,
                        help="token_capacity control's capacity factor")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ExperimentConfig()

    if args.method:
        cfg.method = args.method
    if args.budget:
        cfg.memory.budget_bytes = args.budget
    if args.num_experts:
        cfg.model.num_experts = args.num_experts
    if args.seed:
        cfg.train.seed = args.seed
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.dataset:
        cfg.dataset = args.dataset
        if args.dataset == "cifar100":
            cfg.model.num_classes = 100
    if args.name:
        cfg.experiment_name = args.name
    if args.d_model:
        cfg.model.d_model = args.d_model
    if args.d_ff_shared:
        cfg.model.d_ff_shared = args.d_ff_shared
    if args.patch_size:
        cfg.model.patch_size = args.patch_size
    if args.widths:
        widths = [int(w) for w in args.widths.split(",")]
        cfg.model.d_ff_expert = widths
        cfg.model.num_experts = len(widths)
    if args.admissible:
        cfg.model.admissible_mask = [int(i) for i in args.admissible.split(",")]
    if args.capacity_factor is not None:
        cfg.model.capacity_factor = args.capacity_factor
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.budget_schedule:
        cfg.memory.budget_schedule = args.budget_schedule
    elif not hasattr(cfg.memory, "budget_schedule"):
        cfg.memory.budget_schedule = "static"
    if args.budget_min is not None:
        cfg.memory.budget_min_bytes = args.budget_min
    elif not hasattr(cfg.memory, "budget_min_bytes"):
        cfg.memory.budget_min_bytes = cfg.memory.budget_bytes
    if args.budget_cycle is not None:
        cfg.memory.budget_cycle_steps = args.budget_cycle
    elif not hasattr(cfg.memory, "budget_cycle_steps"):
        cfg.memory.budget_cycle_steps = 1

    budget_desc = f"{cfg.memory.budget_bytes//1024}KB"
    if getattr(cfg.memory, "budget_schedule", "static") != "static":
        budget_desc = (
            f"{getattr(cfg.memory, 'budget_min_bytes', cfg.memory.budget_bytes)//1024}"
            f"-{cfg.memory.budget_bytes//1024}KB/{cfg.memory.budget_schedule}"
        )

    print(f"Running: {cfg.experiment_name} | method={cfg.method} | "
          f"budget={budget_desc} | experts={cfg.model.num_experts} | "
          f"seed={cfg.train.seed}")

    results = run_experiment(cfg)
    print(f"Best test acc: {results['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
