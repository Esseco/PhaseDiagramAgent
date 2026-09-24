"""提供可修改的多精度与 API 限额。"""


def default_budget_rules() -> dict:
    return {
        "total_relative_cost": 20_000.0,
        "cost_unit": "relative_cost",
        "cost_model": {"reference_atoms": 40.0, "atom_exponent": 1.0, "scale": 1.0,
                       "stage_exponents": {"relax_and_feature": 1.2, "deep_search": 1.2,
                                           "dft_single_point": 3.0, "dft_relax": 3.0},
                       "reference_mc_steps": 1.0,
                       "deep_search_cost_basis": "one_mlip_relaxation_per_mc_step"},
        "structure_limits": {"max_atoms": 500, "max_det_H": 64, "max_proxy_cost_per_task": None},
        "stage_limits": {
            "simple_check": {"max_tasks": None, "max_cost": 20.0, "task_cost": 0.05},
            "relax_and_feature": {"max_tasks": 500, "max_cost": 500.0, "task_cost": 1.0},
            "deep_search": {"max_tasks": 100, "max_cost": 4000.0, "task_cost": 1.0},
            "dft_single_point": {"max_tasks": 100, "max_cost": 6000.0, "task_cost": 30.0},
            "dft_relax": {"max_tasks": 2, "max_cost": 15000.0, "task_cost": 900.0},
        },
        "llm_limits": {
            "max_calls": 100,
            "max_input_tokens": 200_000,
            "max_output_tokens": 20_000,
            "max_cost": None,
            "max_calls_per_iteration": 1,
        },
        "minimum_random_exploration_fraction": 0.10,
        "minimum_independent_dft_audit_fraction": 0.05,
    }
