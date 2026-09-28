"""Default user-editable layered-oxide platform configuration."""

from config_layer.defaults.default_budget_rules import default_budget_rules
from config_layer.defaults.layered_oxide_system_config import layered_oxide_system_config
from config_layer.defaults.default_dft_decision_config import default_dft_decision_config
from scientific_layer.bohb.default_bohb_config import default_bohb_config
from decision_layer.strategy.default_round_strategy_config import default_round_strategy_config
from config_layer.defaults.default_mace_committee_config import default_mace_committee_config
from config_layer.defaults.default_supercomputer_config import default_supercomputer_config


def default_layered_search_config(*, boundary=None, phase_references=None) -> dict:
    return {
        "system": layered_oxide_system_config(boundary=boundary, phase_references=phase_references),
        "frozen_parameters": ["system.constraints", "system.branch_schema", "budgets.total_relative_cost",
                              "dft.parameters", "convergence.final_energy_mae_tolerance",
                              "convergence.hull_change_tolerance"],
        "branch_partition_suggestions": {"group_by": ["P", "H", "x"], "status": "suggestion"},
        "generation_actions": {"enabled": ["coverage", "composition", "competing_phase", "tm_ordering", "periodic_extension"], "quotas": {}},
        "calculation": {"enabled_stages": ["simple_check", "relax_and_feature", "deep_search", "dft_single_point", "dft_relax"], "mc_allocator": "agent_tools", "bohb_optional": True, "mlip_version": "mace-mh-1"},
        "budgets": default_budget_rules(),
        "mlip": {"name": "mace-mh-1", "mace_head": "omat_pbe", "model_path": "/data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model", "model_paths": [],
                 "device": "cuda", "main_model_index": 0,
                 "mc_parameters": {"save_relax_traj": False}},
        "mlip_finetune": default_mace_committee_config(),
        "initial_mlip_health_check": {"required_phase_roles": ["endpoint", "intermediate"],
                                      "max_failure_fraction": 0.5,
                                      "max_energy_error_ev_per_atom": None,
                                      "required_before_large_scale_search": True},
        "dft": {"backend": "atomate", "parameter_source": "atomate_defaults",
                "parameters": {}, "independent_audit_fraction": 0.05,
                "selection": {"near_hull_ev_per_atom": 0.10, "single_point_first": True,
                              "max_relax_fraction": 0.10,
                              "agent_threshold_adjustment": {"minimum": 0.05, "maximum": 0.20}}},
        "mc_policy": {
            "initial_ehull_bands": [
                {"max_ehull_ev_per_atom": 0.010, "max_mc_steps": 100, "patience_steps": 8},
                {"max_ehull_ev_per_atom": 0.020, "max_mc_steps": 60, "patience_steps": 6},
                {"max_ehull_ev_per_atom": 0.040, "max_mc_steps": 30, "patience_steps": 4},
            ],
            "exploration_max_mc_steps": 15, "exploration_patience_steps": 3,
            "second_segment_enabled": False,
            "tiers": [
                {"name": "small", "max_mc_steps": 10, "patience_steps": 3, "min_improvement": 0.001},
                {"name": "medium", "max_mc_steps": 30, "patience_steps": 6, "min_improvement": 0.001},
                {"name": "large", "max_mc_steps": 90, "patience_steps": 12, "min_improvement": 0.0005}
            ],
            "max_segments_per_branch": 2, "max_cumulative_cost_per_branch": 200.0,
            "random_exploration_fraction": 0.10, "high_ehull_defer_threshold": 0.040,
            "near_hull_retry_threshold": 0.10,
            "agent_adjustable": {"random_exploration_fraction": [0.10, 0.30],
                                 "dft_near_hull_ev_per_atom": [0.05, 0.20]}
        },
        "convergence": {"hull_change_tolerance": 0.003, "stable_model_update_epochs": 2,
                        "final_energy_mae_tolerance": 0.003,
                        "minimum_recent_dft_checks": 1, "coverage_is_hard_condition": False},
        "reliability": {"require_dft_validation": True, "model_update_requires_validation": True},
        "agent": {"enabled": True, "rule_fallback": True, "allowed_tools": ["generate_branches", "select_candidates", "prepare_dedup_batch", "prepare_local_batch_files", "run_calculation_stage", "allocate_mc_bohb", "select_dft_candidates", "update_mlip", "reevaluate_candidates", "check_convergence", "pause_search", "restart_failed_task", "adjust_strategy"]},
        "dedup": {"budget_limit": 10.0, "resource_limit": {"max_tasks": 100, "max_concurrency": 4},
                  "debug_requires_approval": True},
        "run": {"total_quota": 300, "batch_size": 96, "initial_states_per_branch": 4, "seed": 42},
        "qbc": default_dft_decision_config(),
        "bohb": default_bohb_config(),
        "round_strategy": default_round_strategy_config(),
        "supercomputer": default_supercomputer_config(),
    }
