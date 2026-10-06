"""Keep raw DFT magnetization on transport; assess only on scientific consumption."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from unittest.mock import Mock

from tests.test_dft_spin_acceptance import make_result, make_manager
from scientific_layer.dft.spin_acceptance import apply_dft_spin_standard, spin_standard_passed
from execution_layer.state.reconcile_task_results import reconcile_task_results
from data_layer.ledger.collect_calculation_results import collect_calculation_results
from analysis_layer.feedback.dft_result_products import record_dft_products, export_dft_products
from run.workflow_reply_presentation import format_workflow_reply


def raw_result():
    incoming = make_result(moments=(4.3, 1.2, 0, 0))
    legacy = incoming["outputs"].pop("magnetic_check")
    incoming["outputs"]["magnetic_moments"] = {**legacy, "schema": "vasp-final-site-moments-v1",
        "status": "available", "unit": "mu_B", "final_frame_index": 0}
    incoming["outputs"].update(final_frame_index=0, training_frame_index=0)
    return incoming


def test_raw_transport_receives_anomalous_spins_without_science_filter():
    incoming = raw_result()
    before = deepcopy(incoming)
    state = {"tasks": [{"task_id": "D1", "task_key": "K1", "stage": "dft_single_point", "status": "pending"}]}
    reconciled = reconcile_task_results(state, [incoming])["state"]
    assert reconciled["tasks"][0]["status"] == "completed"
    assert reconciled["tasks"][0]["checks_passed"] is True
    assert reconciled["tasks"][0]["outputs"]["magnetic_moments"]["moments"][1] == 1.2
    assert "spin_state_check" not in reconciled["tasks"][0]["outputs"]
    manager = make_manager()
    collected = collect_calculation_results(manager, "S1", incoming)
    assert collected["status"] == "completed"
    assert collected["phase_record"]["magnetic_moments"]["moments"][1] == 1.2
    assert spin_standard_passed(collected["phase_record"]) is False  # raw is not yet scientifically approved
    assert incoming == before


def test_analysis_preserves_raw_marks_good_bad_and_excludes_bad_task(tmp_path):
    incoming = raw_result()
    manager = make_manager()
    checked = apply_dft_spin_standard(incoming, manager)
    assert checked["outputs"]["magnetic_moments"] == incoming["outputs"]["magnetic_moments"]
    report = checked["outputs"]["magnetic_check"]
    assert report["reasonable_atoms"][0]["element"] == "Fe"
    assert report["anomalous_atoms"][0]["element"] == "Mn"
    predictor = Mock(side_effect=AssertionError("bad spin should not evaluate"))
    state = {}
    record_dft_products(state, incoming, predictor, manager)
    export_dft_products(state, tmp_path)
    predictor.assert_not_called()
    root = Path(next(iter(state["dft_result_exports"].values()))["directory"])
    saved = json.loads((root / "dft_records.json").read_text())
    assert saved[0]["magnetic_moments"] == incoming["outputs"]["magnetic_moments"]
    with (root / "magnetic_moments.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["spin_assessment"] for r in rows] == ["reasonable", "out_of_range", "not_checked", "not_checked"]
    compact = format_workflow_reply({"state": state, "status": "completed"}, tmp_path / "state.json")
    assert "异常 1" in compact and "Mn[1]" not in compact
    reply = format_workflow_reply({"state": state, "status": "completed"}, tmp_path / "state.json", verbose=True)
    assert "Fe 3.5–4.5" in reply and "Mn[1]=1.2" in reply
    assert "magnetic_moments.csv" in reply and "异常 1" in reply


def test_vector_raw_is_retained_even_when_unsupported_for_spin_filter():
    incoming = raw_result()
    raw = incoming["outputs"]["magnetic_moments"]
    raw.update(noncollinear=True, moments=[[0,0,4.3], [0,0,3.9], [0,0,0], [0,0,0]])
    checked = apply_dft_spin_standard(incoming, make_manager())
    assert checked["outputs"]["magnetic_moments"] == raw
    assert checked["outputs"]["spin_state_check"]["status"] == "unknown"
    assert checked["checks_passed"] is False
