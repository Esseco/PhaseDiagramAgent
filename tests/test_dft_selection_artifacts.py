from phase_agent.tools.workflows.branch_review_view import raw_record
import json
from pathlib import Path
from copy import deepcopy

from phase_agent.tools.workflows.dft_selection_artifacts import proposal_artifacts, preparation_artifacts
from phase_agent.runtime.concise_dialogue import approval_reply


def fixture(tmp_path):
    state = {"workspace_layout": {"workspace_root": str(tmp_path)}, "active_model_version": "m1",
             "qbc_candidates": [{"candidate_id": "s1", "branch_id": "b1", "phase": "O3", "x_Na_per_O2": 0.5,
                                  "composition": {"Na": 1, "Fe":2, "O":4}, "predicted_Ehull": 0.01, "ehull_unit": "eV/atom",
                                  "structure_path": "s1.vasp", "qbc": {"energy_per_atom_std": 0.02}}]}
    action = {"tool": "select_dft_candidates", "task_key": "select1", "parameters": {
        "decisions": [{"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "reason": "近凸包校验"},
                      {"candidate_id": "s2", "action": "DEFER", "reason": "已有证据"}],
        "dft_input_preview": {"task_count": 1, "relative_cost": 2}}}
    return state, action


def test_approval_exports_one_json_without_changing_science(tmp_path):
    state, action = fixture(tmp_path)
    before = deepcopy(state)
    text = approval_reply({"raw_action": action}, state)
    artifacts = proposal_artifacts({"raw_action": action}, state)
    assert "DFT 选点详情" in text and "文件名：" in text
    assert "s1.vasp" not in text
    assert state == before
    path = Path(artifacts["json"])
    data = raw_record(json.loads(path.read_text(encoding="utf-8")))
    assert data["summary"]["selected_count"] == 1
    row = data["decisions"][0]
    assert row["phase"] == "O3" and row["Na_fraction"] == 0.5 and row["Ehull"] == 0.01
    assert row["H"] is None and row["training_membership"] == "unknown"
    assert data["submit_authorized"] is False and data["actual"] is None
    assert len(list(path.parent.iterdir())) == 1


def test_prepared_inventory_updates_same_file_and_keeps_plan(tmp_path):
    state, action = fixture(tmp_path)
    artifacts = proposal_artifacts({"raw_action": action, "expected_purpose": "独立验证"}, state)
    result = preparation_artifacts(action, state, [{"task_id": "t1", "stage": "dft_single_point"}],
                                   {"accepted": [action["parameters"]["decisions"][0]], "rejected": []},
                                   {"status": "prepared", "directory": "upload"})
    assert result == artifacts
    path = Path(result["json"])
    data = raw_record(json.loads(path.read_text(encoding="utf-8")))
    assert data["purpose"] == "独立验证" and data["actual"]["task_count"] == 1
    assert data["actual"]["input_preparation"]["directory"] == "upload"
    proposal_artifacts({"raw_action": action}, state)
    assert raw_record(json.loads(path.read_text(encoding="utf-8")))["status"] == "prepared"


def test_failed_preparation_is_not_reported_as_prepared(tmp_path):
    state, action = fixture(tmp_path)
    artifacts = preparation_artifacts(action, state, [], {}, {"status": "rejected", "reason": "missing file"})
    assert raw_record(json.loads(Path(artifacts["json"]).read_text(encoding="utf-8")))["status"] == "input_preparation_failed"
