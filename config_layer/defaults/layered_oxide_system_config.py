"""层状氧化物体系的默认配置。"""

from copy import deepcopy


def layered_oxide_system_config(*, boundary=None, phase_references=None) -> dict:
    config = {
        "system_id": "layered_na_tm_oxide",
        "phase_references": dict(phase_references or {}),
        "mc_full_na_templates": {},
        "species": {"mobile": ["Na"], "framework": ["O"], "substitutional": ["Fe", "Mn"]},
        "occupancy_rules": {"mobile_sites": "Na/vacancy", "substitutional_sites": "Fe/Mn", "integer_occupancy": True},
        "constraints": {"phases": ["O3", "O1", "P3", "OP2"], "TM_ratio": {"Fe": 1, "Mn": 1}},
        "configuration_space": {
            "roles": {"H": "branch", "P": "branch", "x": "branch",
                      "T": "branch", "N": "internal"},
            "fixed_T_source": None, "fixed_values": {},
        },
        "branch_schema": {"fields": ["P", "H", "x", "T"], "internal_search_variables": ["V"]},
        "generation": {"enabled_strategies": ["coverage", "composition", "competing_phase", "tm_ordering", "periodic_extension"], "strategy_seed_offsets": {"coverage": 0, "composition": 10000, "competing_phase": 20000, "tm_ordering": 30000, "periodic_extension": 40000}},
        "calculation_workflow": {
            "stages": [
                {"name": "simple_check", "label": "简单检查", "enabled": True, "max_retries": 1},
                {"name": "relax_and_feature", "label": "弛豫+特征识别", "enabled": True, "max_retries": 1},
                {"name": "deep_search", "label": "深度搜索", "enabled": True, "max_retries": 1},
                {"name": "dft_single_point", "label": "DFT 单点能", "enabled": True, "max_retries": 1},
                {"name": "dft_relax", "label": "DFT 弛豫", "enabled": True, "max_retries": 1},
            ],
            "terminal_statuses": ["completed", "failed", "paused", "not_configured", "cancelled"],
            "active_statuses": ["pending", "running"],
        },
    }
    if boundary is not None:
        config["boundary"] = deepcopy(boundary)
        from scientific_layer.structures.boundary_utils import allowed_phases
        config["constraints"]["phases"] = sorted(allowed_phases(boundary.get("P", config["constraints"]["phases"])))
        config["constraints"]["TM_ratio"] = deepcopy(boundary.get("TM_ratio", config["constraints"]["TM_ratio"]))
    return config
