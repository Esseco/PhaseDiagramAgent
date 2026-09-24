"""BOHB 默认配置；预算只表示 MLIP+MC 搜索投入。"""


def default_bohb_config() -> dict:
    return {
        "method": "agent_relax_hull_tiered_mc",
        "bohb_selection_interface_enabled": False,
        "selection_policy": "relax_hull_uncertainty",
        "relax_structures_per_branch": 3,
        "uncertainty_weight": 0.25,
        "budget_levels": [10, 30, 90],
        "eta": 3,
        "new_candidates_per_iteration": 9,
        "random_fraction": 0.25,
        "minimum_model_observations": 6,
        "good_fraction": 0.25,
        "kernel_bandwidth": 1.0,
        "objective": {
            "name": "group_normalized_min_energy",
            "group_key": "composition_group",
            "value_key": "minimum_energy_per_atom",
            "reference_key": "group_reference_energy_per_atom",
            "scale_key": "group_energy_scale",
            "lower_is_better": True,
        },
        "scope": {
            "mlip_version": None,
            "hull_reference_version": None,
            "candidate_set_version": None,
        },
    }
