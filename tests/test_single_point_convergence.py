from copy import deepcopy

import pytest

from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)


RULES = {
    "hull_change_tolerance": 0.003,
    "stable_model_update_epochs": 1,
    "final_energy_mae_tolerance": 0.01,
    "minimum_recent_dft_checks": 1,
    "phase_competition_gap_ev_per_atom": 0.003,
}


def state(gap=0.002, *, reversed=False):
    rows, comparisons, entries = [], [], []
    for identity, phase, mlip, dft in (
        ("a", "O3", -1.0, -1.0),
        ("b", "P2", -1.0 + gap, -1.0 - gap if reversed else -1.0 + gap),
    ):
        rows.append(
            {
                "task_id": identity,
                "structure_id": identity,
                "model_version": "m1",
                "stage": "dft_single_point",
                "status": "completed",
                "converged": True,
                "training_ready": True,
                "actual_phase": phase,
                "composition": {"Na": 1},
                "atom_count": 1,
                "energy": dft,
                "energy_unit": "eV",
                "forces": [[0.03, 0.04, 0]],
                "forces_unit": "eV/angstrom",
                "mlip_prediction": {
                    "model_version": "m1",
                    "energy": mlip,
                    "energy_unit": "eV",
                    "forces": [[0, 0, 0]],
                    "forces_unit": "eV/angstrom",
                },
            }
        )
        comparisons.append({"task_id": identity, "status": "completed"})
        entries.append(
            {
                "structure_id": identity,
                "phase": phase,
                "composition": {"Na": 1},
                "normalized_total_energy": mlip,
            }
        )
    return {
        "active_model_version": "m1",
        "tasks": [],
        "model_update_epochs": [
            {"model_version": "m1", "hull_change": 0.001, "ground_state_unchanged": True}
        ],
        "dft_dataset_records": rows,
        "dft_mlip_comparisons": comparisons,
        "phase_diagrams": {
            "mlip": {"status": "completed", "model_version": "m1", "entries": entries}
        },
    }


def test_one_round_same_frame_metrics_and_competition_pass():
    result = check_global_convergence(state(), rules=RULES)
    assert result["criteria_satisfied"]
    metrics = result["single_point_validation"]["metrics"]
    assert metrics["forces"]["mae"] == pytest.approx(0.07 / 3)
    assert metrics["forces"]["rmse"] == pytest.approx((0.0025 / 3) ** 0.5)
    assert metrics["dft_max_residual_force_ev_per_angstrom"] == pytest.approx(0.05)
    assert metrics["energy_per_atom"]["mae"] == 0


def test_reversal_fails_even_when_average_energy_error_passes():
    result = check_global_convergence(state(reversed=True), rules=RULES)
    assert result["checks"]["final_frame_dft_calibrated"]
    assert not result["checks"]["competing_phases_agree"]
    assert result["status"] == "continue"


def test_missing_dft_requests_only_missing_representative_without_submitting():
    data = state()
    data["dft_dataset_records"].pop()
    result = check_global_convergence(data, rules=RULES)
    assert result["status"] == "insufficient_evidence"
    assert result["single_point_validation"]["phase_competition"]["required_dft_structure_ids"] == [
        "b",
    ]
    assert data["tasks"] == []


@pytest.mark.parametrize("gap", [0.003, 0.004])
def test_large_gap_requires_no_extra_competition_dft(gap):
    data = state(gap)
    data["dft_dataset_records"].pop()
    result = check_global_convergence(data, rules=RULES)
    assert result["criteria_satisfied"]
    assert result["single_point_validation"]["phase_competition"]["checks"] == []


def test_same_phase_needs_no_ordering_agreement():
    data = state(reversed=True)
    for row in data["phase_diagrams"]["mlip"]["entries"]:
        row["phase"] = "O3"
    assert check_global_convergence(data, rules=RULES)["criteria_satisfied"]


@pytest.mark.parametrize("change", ["old_model", "relaxed_frame", "stale_energy", "bad_labels"])
def test_invalid_competition_evidence_stays_unresolved(change):
    data = state()
    row = data["dft_dataset_records"][-1]
    if change == "old_model":
        row["model_version"] = "m0"
    elif change == "relaxed_frame":
        row["stage"] = "dft_relax"
    elif change == "stale_energy":
        row["mlip_prediction"]["energy"] += 0.01
    else:
        row["checks_passed"] = False
    result = check_global_convergence(data, rules=RULES)
    assert not result["criteria_satisfied"]
    assert "competing_phase_same_frame_dft" in result["missing_evidence"]


def test_explicit_force_and_rmse_limits_are_enforced():
    rules = {**RULES, "final_force_rmse_tolerance": 0.01, "final_energy_rmse_tolerance": 0.001}
    result = check_global_convergence(state(), rules=rules)
    assert not result["checks"]["final_force_rmse_tolerance"]
    assert result["checks"]["final_energy_rmse_tolerance"]


def test_missing_snapshot_and_tied_phases_do_not_pass():
    data = state()
    data.pop("phase_diagrams")
    assert check_global_convergence(data, rules=RULES)["status"] == "insufficient_evidence"
    assert not check_global_convergence(state(0), rules=RULES)["criteria_satisfied"]


def test_legacy_rules_still_work_and_nonfinite_hull_is_rejected():
    rules = deepcopy(RULES)
    rules.pop("phase_competition_gap_ev_per_atom")
    data = {
        "model_update_epochs": [
            {"model_version": "m1", "hull_change": 0.001, "ground_state_unchanged": True}
        ],
        "final_frame_dft_errors": [{"error_ev_per_atom": -0.001}],
    }
    assert check_global_convergence(data, rules=rules)["criteria_satisfied"]
    data["model_update_epochs"][0]["hull_change"] = float("nan")
    assert not check_global_convergence(data, rules=rules)["criteria_satisfied"]


def test_defaults_keep_one_round_and_do_not_invent_force_thresholds():
    config = default_layered_search_config()["convergence"]
    assert config["stable_model_update_epochs"] == 1
    assert config["phase_competition_gap_ev_per_atom"] == 0.003
    assert config["final_force_mae_tolerance"] is None


@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf"), True])
def test_invalid_optional_threshold_cannot_bypass_validation(value):
    result = check_global_convergence(
        state(), rules={**RULES, "phase_competition_gap_ev_per_atom": value}
    )
    assert result["status"] == "needs_confirmed_convergence_rules"
