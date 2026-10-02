"""Budget sweep experiment: accuracy vs peak SRAM across all methods."""

import itertools
import json
from pathlib import Path
from copy import deepcopy

from gatedmoe.util_config_io import ExperimentConfig, LOG_DIR


# Fixed budget, vary batch size to create different memory pressures.
# shared+expert cost scales linearly with batch size:
#   bs=4:  shared=1347KB, s+e=2694KB
#   bs=8:  shared=2179KB, s+e=4358KB
#   bs=16: shared=3843KB, s+e=7686KB
#   bs=32: shared=7171KB, s+e=14342KB
# At budget=8MB:
#   bs=4:  always promotes (2694 << 8000)
#   bs=8:  always promotes (4358 < 8000)
#   bs=16: just promotes (7686 < 8000)
#   bs=32: never promotes (14342 > 8000)
BUDGET_BYTES = 8 * 1024 * 1024  # Fixed 8MB budget
BATCH_SIZES = [4, 8, 12, 16, 24, 32]  # Creates smooth transition
METHODS = ["dense", "learned_top1", "random", "round_robin", "mod", "gatedmoe_base", "gatedmoe_g"]
SEEDS = [42, 123, 456]


def generate_sweep_configs(base_cfg: ExperimentConfig) -> list[ExperimentConfig]:
    """Generate all (method, batch_size, seed) combinations at fixed budget."""
    configs = []
    for method, bs, seed in itertools.product(METHODS, BATCH_SIZES, SEEDS):
        cfg = deepcopy(base_cfg)
        cfg.method = method
        cfg.memory.budget_bytes = BUDGET_BYTES
        cfg.train.batch_size = bs
        cfg.train.seed = seed
        cfg.experiment_name = f"sweep_{method}_bs{bs}_s{seed}"
        configs.append(cfg)
    return configs


def collect_sweep_results(log_dir: str = str(LOG_DIR)) -> list[dict]:
    """Collect all sweep results from saved JSON files."""
    results = []
    log_path = Path(log_dir)
    for exp_dir in sorted(log_path.glob("sweep_*")):
        result_file = exp_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                last = data["epochs"][-1]
                results.append({
                    "method": data["config"]["method"],
                    "budget_bytes": data["config"]["budget_bytes"],
                    "batch_size": data["config"].get("batch_size", 16),
                    "seed": data["config"]["seed"],
                    "best_test_acc": data["best_test_acc"],
                    "final_test_acc": data["final_test_acc"],
                    "peak_sram_bytes": last["peak_sram_bytes"],
                    "flash_to_sram_bytes": last["flash_to_sram_bytes"],
                    "avg_macs": last["avg_macs_per_step"],
                })
    return results
