"""Ablation experiment runner: debt type × num_experts × shared config."""

import itertools
from copy import deepcopy

from gatedmoe.util_config_io import ExperimentConfig


DEBT_TYPES = ["gatedmoe_base", "gatedmoe_g"]  # + "no_debt" handled by round_robin
NUM_EXPERTS = [2, 4, 8]
SEEDS = [42, 123, 456]


def generate_ablation_configs(base_cfg: ExperimentConfig) -> list[ExperimentConfig]:
    """Generate ablation matrix configs."""
    configs = []

    # Debt type × num_experts
    for method, n_exp, seed in itertools.product(DEBT_TYPES, NUM_EXPERTS, SEEDS):
        cfg = deepcopy(base_cfg)
        cfg.method = method
        cfg.model.num_experts = n_exp
        cfg.train.seed = seed
        cfg.experiment_name = f"ablation_{method}_e{n_exp}_s{seed}"
        configs.append(cfg)

    # No-debt baseline (round robin) for comparison
    for n_exp, seed in itertools.product(NUM_EXPERTS, SEEDS):
        cfg = deepcopy(base_cfg)
        cfg.method = "round_robin"
        cfg.model.num_experts = n_exp
        cfg.train.seed = seed
        cfg.experiment_name = f"ablation_nodebtrr_e{n_exp}_s{seed}"
        configs.append(cfg)

    return configs
