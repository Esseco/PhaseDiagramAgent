"""提供一份可直接修改的搜索配置。"""

from __future__ import annotations

from typing import Any

from config_layer.defaults.default_budget_rules import default_budget_rules
from scientific_layer.bohb.default_bohb_config import default_bohb_config
from decision_layer.strategy.default_round_strategy_config import default_round_strategy_config
from config_layer.defaults.layered_oxide_system_config import layered_oxide_system_config
from config_layer.defaults.default_mace_committee_config import default_mace_committee_config
from config_layer.defaults.default_supercomputer_config import default_supercomputer_config


def default_run_config() -> dict[str, Any]:
    """返回不包含机器路径和计算后端的默认配置。"""
    config = {
        "system_config": layered_oxide_system_config(),
        "structure_directory": "outputs/candidate_structures",
        "state_path": "outputs/search_state.json",
        "ledger_path": "outputs/phase_data.json",
        "branch_energy_pool_ledger_path": "outputs/branch_energy_pools.json",
        "phase_diagram_directory": "outputs/phase_diagrams",
        "total_quota": 600,
        "batch_size": 250,
        "initial_states_per_branch": 3,
        "seed": 42,
        "mlip": {
            "name": "mace-mh-1",
            "mace_head": "omat_pbe",
            "environment": "py-mace",
            "model_path": None,
            "model_paths": [],
            "device": "cuda",
            "main_model_index": 0,
            "mc_parameters": {"save_relax_traj": False},
        },
        "mlip_finetune": default_mace_committee_config(),
        "qbc": {
            "selection_mode": "agent",
            "minimum_models": 4,
            "energy_threshold": 0.02,
            "force_rms_threshold": 0.15,
            "force_max_threshold": 0.5,
            "output_path": "outputs/qbc/qbc_results.json",
        },
        "dft": {"backend": "atomate", "parameter_source": "atomate_defaults", "parameters": {}},
        "deepseek": {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "api_key": None,
            "max_tokens": 800,
        },
        "mc_policy": {
            "tiers": [
                {"name": "small", "max_mc_steps": 10, "patience_steps": 3, "min_improvement": 0.001},
                {"name": "medium", "max_mc_steps": 30, "patience_steps": 6, "min_improvement": 0.001},
                {"name": "large", "max_mc_steps": 90, "patience_steps": 12, "min_improvement": 0.0005},
            ],
            "max_segments_per_branch": 3, "max_cumulative_cost_per_branch": 90.0,
            "random_exploration_fraction": 0.10, "high_ehull_defer_threshold": 0.30,
            "near_hull_retry_threshold": 0.10,
        },
        "energy_conventions": {"oxygen_reference_unit": "eV/O2", "model_error_unit": "eV/atom",
                               "voltage_references": {}},
        "budgets": default_budget_rules(),
        "task_versions": {
            "simple_check": None,
            "relax_and_feature": None,
            "deep_search": None,
            "dft_single_point": "atomate",
            "dft_relax": "atomate",
        },
        "generation_metrics": {
            "coverage_gap": 1.0,
            "strategy_efficiency": {},
        },
        "generation_options": {
            "cost_config": {
                "reference_atoms": 100,
                "atom_exponent": 1.0,
                "planned_search_budget": 10,
            },
            "cost_budget": None,
            "selection_config": {
                "max_per_framework": 8,
                "max_per_parent_branch": 2,
                "random_fraction": 0.2,
            },
        },
        "supercomputer": default_supercomputer_config(),
    }
    config["bohb"] = default_bohb_config()
    config["bohb"]["budget_limits"] = config["budgets"]
    config["round_strategy"] = default_round_strategy_config()
    return config
