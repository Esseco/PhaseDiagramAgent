"""Default four-member MACE committee copied from the reviewed 1.py settings."""

from copy import deepcopy


_MEMBERS = [
    {
        "loss": "huber",
        "energy_weight": 10,
        "forces_weight": 10,
        "stress_weight": 1,
        "lr": 1.0e-3,
        "seed": 2026,
    },
    {
        "loss": "stress",
        "energy_weight": 1,
        "forces_weight": 10,
        "stress_weight": 1,
        "lr": 7.5e-4,
        "seed": 2027,
    },
    {
        "loss": "universal",
        "energy_weight": 10,
        "forces_weight": 3,
        "stress_weight": 1,
        "lr": 5.0e-4,
        "seed": 2028,
    },
    {
        "loss": "l1l2energyforces",
        "energy_weight": 5,
        "forces_weight": 5,
        "stress_weight": 0,
        "lr": 1.25e-3,
        "seed": 2029,
    },
]


def default_mace_committee_config(*, foundation_model=None) -> dict:
    """Return four reproducible members; member zero is always the main model."""
    return {
        "enabled": False,
        "output_directory": "outputs/mace_committee",
        "seed": 2026,
        "cross_validation": {"folds": 5},
        "group_keys": ["branch_id", "framework_id"],
        "split": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "labels": {
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "config_type": "Default",
            "include_stress": True,
            "stress_key": "REF_stress",
            "stress_unit": "kbar",
        },
        "training": {
            "foundation_model": foundation_model,
            "minimum_new_dft_records": 10,
            "multiheads_finetuning": False,
            "E0s": "estimated",
            "weight_decay": 0.0,
            "ema": True,
            "ema_decay": 0.999,
            "amsgrad": True,
            "scaling": "rms_forces_scaling",
            "clip_grad": 1.0,
            "batch_size": 4,
            "max_num_epochs": 200,
            "patience": 20,
            "default_dtype": "float64",
            "device": "cuda",
        },
        "committee": [
            {**deepcopy(member), "bootstrap_seed": member["seed"]} for member in _MEMBERS
        ],
        "main_model_index": 0,
        "qbc_member_count": 4,
        "activation": {"requires_separate_approval": True, "pause_on_refresh_anomaly": True},
        "validation": {
            "max_energy_mae": None,
            "max_force_rmse": None,
            "max_energy_mae_increase": None,
            "max_force_rmse_increase": None,
            "minimum_distinguishable_energy_mae_gain": 0.0,
            "max_critical_failure_fraction": None,
            "max_near_hull_ranking_reversals": None,
        },
        "refresh_validation": {
            "max_failure_fraction": None,
            "max_near_hull_ranking_reversals": None,
            "max_energy_mae_ev_per_atom": None,
        },
    }
