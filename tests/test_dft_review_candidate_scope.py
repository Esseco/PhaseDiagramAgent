from copy import deepcopy

import pytest

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.tools.workflows.create_dft_selection_handler import (
    _state_candidates,
    create_dft_selection_handler,
)


def candidate(name):
    return {
        "candidate_id": name,
        "structure_id": name,
        "branch_id": "branch-" + name,
        "atom_count": 10,
        "predicted_Ehull": 0.02,
        "qbc": {"status": "completed"},
    }


@pytest.mark.parametrize("decision", ["DEFER", "REJECT"])
def test_nonexecuting_decisions_remain_available_for_approved_validation(
    decision, monkeypatch, tmp_path
):
    # Plot validity is covered separately; this test isolates decision scope.
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_plot.require_dft_selection_plot",
        lambda *a, **k: None,
    )
    state = {
        "qbc_candidates": [candidate("selected"), candidate("later"), candidate("other")],
        "tasks": [],
        "pending_tasks": [],
    }
    action = {
        "target_ids": ["selected"],
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "selected", "action": "DFT_SINGLE_POINT", "reason": "verify"},
                {"candidate_id": "later", "action": decision, "reason": "later"},
            ]
        },
    }
    selected_path = tmp_path / "selected.vasp"
    selected_path.write_text(
        "selected\n1\n4 0 0\n0 4 0\n0 0 4\nNa\n1\nDirect\n0 0 0\n", encoding="utf-8"
    )
    state["qbc_candidates"][0]["structure_path"] = str(selected_path)
    import hashlib

    digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    state["active_model_version"] = "m"
    state["tasks"].append(
        {
            "task_id": "monitor",
            "status": "completed",
            "model_version": "m",
            "outputs": {
                "structure_checksum": digest,
                "extrapolation_monitor": {
                    "final_feature_difference": 0.2,
                    "extrapolation_failure": False,
                    "failures": [],
                },
            },
        }
    )
    original = deepcopy(state)
    available = _state_candidates(state, None, action)
    assert {r["candidate_id"] for r in available} == {"selected", "later"}
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_artifacts.preparation_artifacts",
        lambda *a, **k: {},
    )
    config = {
        "qbc": default_dft_decision_config(),
        "budgets": default_budget_rules(),
        "mlip": {"version": "m"},
        "dft": {"parameters": {}},
    }
    result = create_dft_selection_handler()(
        action=action,
        context={"event_state": state, "effective_config": config, "config_version": "v"},
    )
    assert result["validation"]["valid"]
    assert [r["structure_id"] for r in result["tasks"]] == ["selected"]
    assert state == original


def test_selected_candidate_outside_target_is_not_added_to_execution_scope():
    action = {
        "target_ids": ["selected"],
        "parameters": {
            "decisions": [
                {"candidate_id": "outside", "action": "DFT_SINGLE_POINT", "reason": "verify"}
            ]
        },
    }
    rows = _state_candidates(
        {"qbc_candidates": [candidate("selected"), candidate("outside")]}, None, action
    )
    assert {r["candidate_id"] for r in rows} == {"selected"}


def test_unknown_deferred_candidate_is_not_fabricated():
    action = {
        "target_ids": ["selected"],
        "parameters": {
            "decisions": [{"candidate_id": "unknown", "action": "DEFER", "reason": "later"}]
        },
    }
    rows = _state_candidates({"qbc_candidates": [candidate("selected")]}, None, action)
    assert {r["candidate_id"] for r in rows} == {"selected"}


@pytest.fixture
def geometry_handler_case(tmp_path, monkeypatch):
    import hashlib
    import importlib

    module = importlib.import_module("phase_agent.tools.workflows.create_dft_selection_handler")
    # Geometry checks are independent of the separately tested plot gate.
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_plot.require_dft_selection_plot",
        lambda *a, **k: None,
    )
    path = tmp_path / "selected.vasp"
    path.write_text(
        "selected\n1.0\n4 0 0\n0 4 0\n0 0 4\nNa\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = candidate("selected")
    row.update(
        structure_path=str(path),
        structure_sha256=digest,
        dft_sampling_evidence={"status": "available", "geometry_key": digest},
        geometry_optimization_evidence={
            "source_task_id": "search-old",
            "residual_force_ev_per_angstrom": 0.2,
            "force_tolerance_ev_per_angstrom": 0.05,
        },
    )
    state = {
        "qbc_candidates": [row],
        "tasks": [],
        "pending_tasks": [],
        "effective_decisions": {},
        "budget_reservations": {},
        "reserved_relative_cost": 0,
        "reserved_dft_cost": 0,
    }
    state["active_model_version"] = "m"
    state["tasks"].append(
        {
            "task_id": "monitor",
            "status": "completed",
            "model_version": "m",
            "outputs": {
                "structure_checksum": digest,
                "extrapolation_monitor": {
                    "final_feature_difference": 0.2,
                    "extrapolation_failure": False,
                    "failures": [],
                },
            },
        }
    )
    action = {
        "tool": "select_dft_candidates",
        "target_ids": ["selected"],
        "parameters": {
            "decisions": [
                {"candidate_id": "selected", "action": "DFT_SINGLE_POINT", "reason": "verify"}
            ]
        },
    }
    context = {
        "event_state": state,
        "config_version": "v",
        "effective_config": {
            "qbc": default_dft_decision_config(),
            "budgets": default_budget_rules(),
            "mlip": {"version": "m"},
            "dft": {"parameters": {}},
        },
    }
    reservations = []
    reserve = module.reserve_dft_actions

    def reserve_probe(working_state, decisions, **kwargs):
        reservations.append(deepcopy(decisions))
        return reserve(working_state, decisions, **kwargs)

    monkeypatch.setattr(module, "reserve_dft_actions", reserve_probe)
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_artifacts.preparation_artifacts",
        lambda *a, **k: {},
    )
    return state, action, context, path, digest, reservations


@pytest.mark.parametrize("dft_action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
@pytest.mark.parametrize(
    "failure",
    ["evidence_mismatch", "candidate_mismatch", "missing_file", "missing_path", "late_mismatch"],
)
def test_geometry_prevalidation_rejects_before_any_reservation(
    geometry_handler_case, dft_action, failure
):
    state, action, context, path, digest, reservations = geometry_handler_case
    action["parameters"]["decisions"][0]["action"] = dft_action
    row = state["qbc_candidates"][0]
    if failure == "evidence_mismatch":
        path.write_bytes(path.read_bytes() + b"\n")
    elif failure == "candidate_mismatch":
        row["structure_sha256"] = "a" * 64
    elif failure == "missing_file":
        row["structure_path"] = str(path.parent / "missing.vasp")
    elif failure == "missing_path":
        row.pop("structure_path")
    else:
        later = deepcopy(row)
        later.update(candidate_id="later", structure_id="later")
        later["structure_sha256"] = "a" * 64
        state["qbc_candidates"].append(later)
        action["target_ids"].append("later")
        action["parameters"]["decisions"].append(
            {"candidate_id": "later", "action": "DFT_SINGLE_POINT", "reason": "verify"}
        )
    before = deepcopy(state)
    result = create_dft_selection_handler()(action=action, context=context)
    assert result["status"] == "rejected"
    assert result["tasks"] == []
    assert result["state"] == before
    assert state == before
    assert reservations == []
    assert result["state"]["budget_reservations"] == {}
    assert result["state"]["reserved_relative_cost"] == 0
    assert result["state"]["reserved_dft_cost"] == 0


@pytest.mark.parametrize("dft_action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
@pytest.mark.parametrize("old_has_geometry", [False, True])
def test_geometry_prevalidation_saves_sha_without_overwriting_completed_task(
    geometry_handler_case, dft_action, old_has_geometry
):
    import hashlib

    state, action, context, path, digest, reservations = geometry_handler_case
    action["parameters"]["decisions"][0]["action"] = dft_action
    old_key = f"v:selected:{dft_action}"
    if old_has_geometry:
        old_key += ":geometry:" + hashlib.sha256(b"old geometry").hexdigest()
    old_task = {
        "task_id": "DFT-" + hashlib.sha256(old_key.encode()).hexdigest()[:12],
        "task_key": old_key,
        "structure_id": "selected",
        "status": "completed",
        "energy": -10.0,
    }
    state["tasks"].append(old_task)
    state["effective_decisions"][old_key] = {"status": "completed"}
    before = deepcopy(state)
    result = create_dft_selection_handler()(action=action, context=context)
    assert result["status"] == "completed"
    assert len(result["tasks"]) == 1
    task = result["tasks"][0]
    assert task["input_structure_sha256"] == digest
    assert task["structure_path"] == str(path)
    assert task["task_key"] == f"v:selected:{dft_action}:geometry:" + digest
    assert task["task_id"] != old_task["task_id"]
    assert result["state"]["tasks"] == before["tasks"] + [task]
    assert len({r["task_id"] for r in result["state"]["tasks"]}) == len(before["tasks"]) + 1
    assert result["state"]["effective_decisions"][old_key] == {"status": "completed"}
    assert result["state"]["budget_reservations"][task["task_key"]]["status"] == "reserved"
    assert len(reservations) == 1
    assert state == before


@pytest.mark.parametrize("dft_action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
@pytest.mark.parametrize("has_geometry", [False, True])
def test_geometry_prevalidation_rejects_existing_task_id_before_reservation(
    geometry_handler_case, dft_action, has_geometry
):
    import hashlib

    state, action, context, path, digest, reservations = geometry_handler_case
    action["parameters"]["decisions"][0]["action"] = dft_action
    key = f"v:selected:{dft_action}"
    if has_geometry:
        key += ":geometry:" + digest
    else:
        state["qbc_candidates"][0].pop("dft_sampling_evidence")
    state["tasks"].append(
        {
            "task_id": "DFT-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            "task_key": "historical-key",
            "structure_id": "selected",
            "status": "completed",
        }
    )
    before = deepcopy(state)
    result = create_dft_selection_handler()(action=action, context=context)
    assert result["status"] == "rejected"
    assert result["tasks"] == []
    assert result["state"] == before
    assert state == before
    assert reservations == []


@pytest.mark.parametrize("dft_action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
def test_geometry_prevalidation_rehashes_legacy_input_with_available_path(
    geometry_handler_case, dft_action
):
    state, action, context, path, digest, reservations = geometry_handler_case
    action["parameters"]["decisions"][0]["action"] = dft_action
    state["qbc_candidates"][0].pop("dft_sampling_evidence")
    before = deepcopy(state)
    result = create_dft_selection_handler()(action=action, context=context)
    assert result["status"] == "completed"
    assert result["tasks"][0]["task_key"] == f"v:selected:{dft_action}"
    assert result["tasks"][0]["input_structure_sha256"] == digest
    assert len(reservations) == 1
    assert state == before


def test_recovery_dft_task_uses_failed_model_not_mh1(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.tools.workflows import validate_dft_agent_decisions as validator
    from phase_agent.tools.workflows import dft_selection_plot

    path = tmp_path / "frame.vasp"
    path.write_text("selected\n1\n4 0 0\n0 4 0\n0 0 4\nNa\n1\nDirect\n0 0 0\n", encoding="utf8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = candidate("picked")
    row.update(structure_path=str(path), structure_sha256=digest)
    cycle = {
        "status": "awaiting_reviewed_dft_labels",
        "invalid_model_version": "failed-epoch",
        "selected": [{"candidate_id": "picked", "structure_sha256": digest}],
    }
    state = {
        "qbc_candidates": [row],
        "active_model_version": "mace-mh-1",
        "tasks": [
            {
                "task_id": "monitor",
                "status": "completed",
                "model_version": "mace-mh-1",
                "outputs": {
                    "structure_checksum": digest,
                    "extrapolation_monitor": {
                        "final_feature_difference": 0.2,
                        "extrapolation_failure": False,
                        "failures": [],
                    },
                },
            }
        ],
        "pending_tasks": [],
        "extrapolation_recovery_cycle": cycle,
    }
    monkeypatch.setattr(validator, "_is_direct_recovery_selection", lambda state, ids: True)
    monkeypatch.setattr(dft_selection_plot, "require_dft_selection_plot", lambda *a, **kw: None)
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "picked", "action": "DFT_SINGLE_POINT", "reason": "representative"}
            ]
        },
    }
    config = {
        "qbc": default_dft_decision_config(),
        "budgets": default_budget_rules(),
        "mlip": {"version": "mace-mh-1"},
        "dft": {"parameters": {}},
    }
    result = create_dft_selection_handler()(
        action=action,
        context={"event_state": state, "effective_config": config, "config_version": "v"},
    )
    assert result["validation"]["valid"]
    assert result["tasks"][0]["model_version"] == "failed-epoch"
