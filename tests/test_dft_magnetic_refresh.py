"""Regression for supplemented DFT evidence, cached phase ordering and concise round replies."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.test_dft_spin_acceptance import make_manager, make_result
from tests.test_dft_magnetic_data_flow import raw_result
from phase_agent.science.dft.spin_acceptance import apply_dft_spin_standard, spin_standard_passed
from phase_agent.tools.state.dft_result_refresh import validated_magnetic_refresh
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def preserve_phase(state, *args, **kwargs):
    return state, {}


def test_scope_final_phase_beats_system_name_and_prior_report():
    incoming = make_result(moments=(1, 1, 0, 0))
    incoming["outputs"].update(actual_phase="spinel", phase_identification={"status": "identified", "phase": "spinel"})
    checked = apply_dft_spin_standard(incoming, make_manager())
    assert checked["checks_passed"] is True
    assert checked["outputs"]["spin_state_check"]["status"] == "not_applicable"


def test_unresolved_custom_scope_is_unknown_not_exempt():
    incoming = raw_result()
    incoming["outputs"].pop("actual_phase")
    incoming["outputs"]["phase_identification"] = {"status": "unknown"}
    manager = make_manager()
    manager.data["system_config"]["system_id"] = "custom-folder-system"
    checked = apply_dft_spin_standard(incoming, manager)
    assert checked["checks_passed"] is False
    assert checked["outputs"]["spin_state_check"]["status"] == "unknown"
    stale = {**incoming["outputs"], "actual_phase": "O3", "spin_state_check": {"applies": False, "status": "not_applicable"}}
    assert spin_standard_passed(stale) is False


def test_phase_before_gate_and_unchanged_phase_not_reclassified(tmp_path):
    manager = make_manager()
    manager.stage_labels = {}
    manager.data["system_config"]["system_id"] = "custom-system"
    incoming = raw_result()
    incoming["outputs"].pop("actual_phase")
    incoming["outputs"].pop("phase_identification")

    def classify(state, *args, **kwargs):
        for task in state["tasks"]:
            task["outputs"].update(actual_phase="O3", phase_identification={"status": "identified", "phase": "O3"})
        return state, {}

    with patch("phase_agent.tools.workflows.apply_scientific_feedback.ensure_phase_identification", side_effect=classify),\
         patch("phase_agent.tools.workflows.apply_scientific_feedback.coverage", return_value={}):
        first = apply_scientific_feedback({}, [incoming], manager=manager, phase_diagram_directory=tmp_path)
    saved = first["state"]["dft_dataset_records"][0]
    assert saved["actual_phase"] == "O3"
    assert saved["spin_state_check"]["status"] == "rejected"
    assert not first["state"].get("new_dft_records")


@pytest.mark.parametrize("key,value", [("energy", -11), ("forces", [[1, 0, 0]] * 4), ("final_frame_index", 2)])
def test_changed_scientific_labels_not_metadata_refresh(key, value):
    prior = make_result(moments=None)
    incoming = make_result()
    incoming["outputs"][key] = value
    if key not in prior["outputs"]:
        prior["outputs"][key] = 0
    check = validated_magnetic_refresh(prior, incoming)
    assert check["status"] == "rejected" and key in check["reason"]
    state = {"tasks": [prior], "processed_task_ids": ["D1"]}
    output = reconcile_task_results(state, [incoming])
    assert output["state"]["tasks"] == [prior]
    assert output["reconciled"][0]["status"] == "rejected"


def test_recomputed_lattice_angles_are_not_geometry_changes():
    prior = raw_result()
    incoming = deepcopy(prior)
    lattice = incoming["outputs"]["structure"]["lattice"]
    lattice["alpha"] += 2e-14
    incoming["outputs"]["magnetic_moments"]["moments"][0] += .01
    checked = validated_magnetic_refresh(prior, incoming)
    assert checked["status"] == "refresh"
    assert checked["result"]["outputs"]["structure"] == prior["outputs"]["structure"]


@pytest.mark.parametrize("change", ["matrix", "coordinate", "species", "property"])
def test_real_structure_changes_remain_rejected(change):
    prior = raw_result()
    incoming = deepcopy(prior)
    structure = incoming["outputs"]["structure"]
    if change == "matrix":
        structure["lattice"]["matrix"][0][0] += 1e-12
    elif change == "coordinate":
        structure["sites"][0]["abc"][0] += 1e-12
    elif change == "species":
        structure["sites"][0]["species"][0]["element"] = "Na"
    else:
        structure["sites"][0]["properties"]["new_property"] = 1
    checked = validated_magnetic_refresh(prior, incoming)
    assert checked["status"] == "rejected"
    assert checked["reason"] == "dft_refresh_structure_mismatch"


def test_same_task_supplement_refreshes_once_no_charge_or_duplicate(tmp_path):
    manager = make_manager()
    manager.stage_labels = {}
    module = "phase_agent.tools.workflows.apply_scientific_feedback."
    collector = Mock(return_value={"phase_record": None})
    evaluator = Mock(return_value={"model_version": "m1", "energy": -9, "energy_unit": "eV",
                                   "forces": [[0,0,0]]*4, "forces_unit": "eV/angstrom"})
    prior = make_result(moments=None)
    with patch(module + "collect_calculation_results", collector),\
         patch(module + "ensure_phase_identification", side_effect=preserve_phase),\
         patch(module + "coverage", return_value={}):
        first = apply_scientific_feedback({}, [prior], manager=manager, phase_diagram_directory=tmp_path)
        state = first["state"]
        state["processed_task_ids"] = ["D1"]
        state["budget_usage"] = {"total_relative_cost": 30}
        second_incoming = make_result()
        second_incoming["actual_cost"] = 900
        reconciled = reconcile_task_results(state, [second_incoming])
        assert reconciled["reconciled"][0]["status"] == "metadata_refreshed"
        assert reconciled["state"]["tasks"][0]["actual_cost"] == 30
        second = apply_scientific_feedback(reconciled["state"], [second_incoming], manager=manager,
                    phase_diagram_directory=tmp_path, final_frame_mlip_evaluator=evaluator)
        third = apply_scientific_feedback(second["state"], [second_incoming], manager=manager,
                    phase_diagram_directory=tmp_path, final_frame_mlip_evaluator=evaluator)
    evaluator.assert_called_once()  # First eligible prediction; unchanged retries reuse it.
    assert second["metadata_refreshed_task_ids"] == ["D1"]
    assert third["metadata_refreshed_task_ids"] == []
    for key in ("cost_history", "dft_dataset_records", "dft_training_records", "new_dft_records", "dft_mlip_comparisons"):
        assert len(third["state"][key]) == 1
    assert third["state"]["budget_usage"] == {"total_relative_cost": 30}
    assert third["state"]["dft_dataset_records"][0]["spin_state_check"]["status"] == "passed"
    assert third["state"]["dft_training_records"][0]["checks_passed"] is True
    assert collector.call_count == 2


def test_partial_reply_missing_magnetism_is_explicit_and_round_scoped():
    old = make_result(moments=None)
    state = {"tasks": [old], "dft_dataset_records": [{"task_id": "another-round", "spin_state_check": {"applies": True, "status": "rejected"}}]}
    question = {"recovered_task_ids": ["D1"], "scope": {"model_version": "m1"},
                "recovered_tasks": 1, "expected_tasks": 2, "recovery_ratio": .5, "pending_task_ids": ["D2"]}
    reply = format_workflow_reply({"status": "awaiting_dft_recovery_decision", "state": state,
                                  "dft_recovery_question": question}, "state.json")
    assert "缺少磁矩 1" in reply and "未评估 1" in reply
    assert "重新提取" in reply and "还要继续回收" in reply
    assert "异常 0" in reply and "异常 1" not in reply


def test_processed_dft_collector_accepts_supplement_with_checked_marker(tmp_path):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.remote.integrity import file_checksum
    incoming = make_result()
    taskdir = tmp_path / "batch" / "00000-D1"
    taskdir.mkdir(parents=True)
    (taskdir / "task.json").write_text(json.dumps({"task_id": "D1", "task_key": "K1", "model_version": "m1"}))
    path = taskdir / "result.json"
    path.write_text(json.dumps(incoming))
    (taskdir / "task.finished.json").write_text(json.dumps({"task_id": "D1", "task_key": "K1", "model_version": "m1",
                                                          "status": "completed", "result_checksum": file_checksum(path)}))
    prior = make_result(moments=None)
    prior.update(input_path=str(taskdir / "task.json"), result_path=str(path))
    state = {"tasks": [prior], "processed_task_ids": ["D1"]}
    runner = RemoteBatchRunner(tmp_path / "unused", worker_command=[])
    collected = runner.collect_results_with_report(state)
    assert len(collected["results"]) == 1 and collected["report"]["metadata_refresh_count"] == 1
    state = reconcile_task_results(state, collected["results"])["state"]
    assert runner.collect_results_with_report(state)["results"] == []


def test_real_ledger_refresh_retains_result_id_and_revokes_science(tmp_path):
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.science.training.prepare_mace_finetune import prepare_mace_finetune
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Fe": 1, "Mn": 1}})
    branch = manager.add_branch(P="O3", H=h, x=0, T=["Fe", "Mn"], composition={"Fe": 1, "Mn": 1, "O": 2})
    structure_id = manager.add_structure(branch_id=branch, arrangement={}, composition={"Fe": 1, "Mn": 1, "O": 2})
    good = make_result()
    good["structure_id"] = structure_id
    module = "phase_agent.tools.workflows.apply_scientific_feedback."
    with patch(module + "ensure_phase_identification", side_effect=lambda state, *a, **kw: (deepcopy(state), {})):
        first = apply_scientific_feedback({}, [good], manager=manager, phase_diagram_directory=tmp_path, active_model_version="m1")
        first_id = first["state"]["phase_records"][0]["record_id"]
        bad = deepcopy(good)
        bad["outputs"]["magnetic_check"]["moments"] = [1, 1, 0, 0]
        second = apply_scientific_feedback(first["state"], [bad], manager=manager, phase_diagram_directory=tmp_path, active_model_version="m1")
        again = apply_scientific_feedback(second["state"], [bad], manager=manager, phase_diagram_directory=tmp_path, active_model_version="m1")
    history = manager.data["structures"][structure_id]["stage_history"]["dft_single_point"]
    assert len(history) == 1 and history[0]["result_id"] == first_id
    assert history[0]["metadata"]["checks_passed"] is False
    assert again["state"]["feedback_processed_task_ids"] == ["D1"]
    assert len(again["state"]["phase_records"]) == len(again["state"]["cost_history"]) == 1
    assert again["state"]["phase_records"][0]["checks_passed"] is False
    assert not again["state"]["new_dft_records"]
    assert again["metadata_refreshed_task_ids"] == []
    dataset = prepare_mace_finetune(again["state"]["dft_training_records"], tmp_path / "train", config={})
    assert dataset["accepted"] == 0


def test_cached_final_phase_reused_on_supplement(tmp_path):
    from pymatgen.core import Structure, Lattice
    manager = make_manager()
    manager.stage_labels = {}
    incoming = raw_result()
    structure = Structure(Lattice.cubic(4), ["Na", "Fe", "Mn", "O", "O"],
                          [[0,0,0],[.5,.5,0],[0,.5,.5],[.5,0,.5],[.25,.25,.25]])
    incoming["outputs"].update(structure=structure.as_dict(), composition=structure.composition.get_el_amt_dict(),
                               atom_count=5, forces=[[0,0,0]]*5,
                               magnetic_moments={"elements": ["Na","Fe","Mn","O","O"], "moments": [0,4.3,1.2,0,0], "final_frame_index": 0})
    incoming["outputs"].pop("actual_phase")
    incoming["outputs"].pop("phase_identification")
    module = "phase_agent.tools.workflows.apply_scientific_feedback."
    with patch("phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast", return_value={"phase": "O3", "method": "test"}) as classify,\
         patch(module + "collect_calculation_results", return_value={"phase_record": None}),\
         patch(module + "coverage", return_value={}):
        first = apply_scientific_feedback({}, [incoming], manager=manager, phase_diagram_directory=tmp_path)
        incoming["outputs"]["magnetic_moments"]["moments"][2] = 3.9
        second = apply_scientific_feedback(first["state"], [incoming], manager=manager, phase_diagram_directory=tmp_path)
        assert classify.call_count == 1
    assert second["state"]["dft_dataset_records"][0]["spin_state_check"]["status"] == "passed"


def test_eligible_metadata_refresh_reuses_prediction_and_not_requeues_consumed_training():
    from phase_agent.analysis.feedback.dft_result_products import record_dft_products
    predictor = Mock(return_value={"model_version": "m1", "energy": -9, "energy_unit": "eV",
                                  "forces": [[0,0,0]]*4, "forces_unit": "eV/angstrom"})
    state = {}
    incoming = make_result()
    record_dft_products(state, incoming, predictor, make_manager())
    original = deepcopy(state["dft_mlip_comparisons"])
    state["new_dft_records"] = []  # Already consumed by a prior model update.
    incoming["outputs"]["magnetic_check"]["moments"][0] = 4.1
    record_dft_products(state, incoming, predictor, make_manager(), refresh=True)
    predictor.assert_called_once()
    assert state["dft_mlip_comparisons"] == original
    assert state["dft_dataset_records"][0]["mlip_prediction"]["energy"] == -9
    assert state["new_dft_records"] == []
    assert len(state["dft_training_records"]) == 1


def test_supplement_preserves_non_spin_quality_failure():
    old = apply_dft_spin_standard(make_result(moments=None), make_manager())
    old["checks_passed_before_spin"] = False
    old["quality_rejection_reasons"].append("invalid_force_labels")
    fresh = make_result()
    merged = validated_magnetic_refresh(old, fresh)["result"]
    checked = apply_dft_spin_standard(merged, make_manager())
    assert checked["outputs"]["spin_state_check"]["status"] == "passed"
    assert checked["checks_passed"] is False


def test_all_dft_points_revoked_does_not_leave_old_current_csv(tmp_path):
    import csv
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from tests.test_na_eform_phase_csv import record
    rows = [record("left", 0, -3), record("right", 1, -4)]
    for row in rows:
        row["energy_method"] = "dft"
        row["source_version"] = "vasp"
    first = update_phase_diagram(rows, output_directory=tmp_path)["diagrams"]["dft"]
    original = Path(first["archive_csv_path"]).read_bytes()
    for row in rows:
        row["checks_passed"] = False
    second = update_phase_diagram(rows, output_directory=tmp_path)["diagrams"]["dft"]
    assert second["status"] == "unknown"
    with Path(second["csv_path"]).open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert Path(first["archive_csv_path"]).read_bytes() == original
    timestamp = Path(second["csv_path"]).stat().st_mtime_ns
    repeated = update_phase_diagram(rows, output_directory=tmp_path)["diagrams"]["dft"]
    assert Path(repeated["csv_path"]).stat().st_mtime_ns == timestamp
