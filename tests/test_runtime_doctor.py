import json
from pathlib import Path

import pytest

from phase_agent.runtime import project_diagnostics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8-sig")


def project(root):
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )

    runtime = root / "agent_runtime.json"
    write(
        runtime,
        {
            "state_path": "workflow_state/state.json",
            "ledger_path": "workflow_state/ledgers/phase_data.json",
            "config_session_path": "parameters/config_session.json",
        },
    )
    config = default_layered_search_config(boundary={"P": ["O3"], "TM_ratio": {"Ni": 1}})
    config["system"]["phase_references"] = {"O3": "structures/O3.vasp"}
    write(root / "parameters/config_session.json", {"status": "draft", "config": config})
    return runtime


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_new_project_diagnosis_does_not_require_state_or_create_files(tmp_path):
    runtime = project(tmp_path)
    before = files(tmp_path)
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["status"] == "configuration_required"
    assert report["sections"]["state"]["data"]["status"] == "not_created"
    assert report["sections"]["references"]["data"]["missing_phases"] == ["O3"]
    assert report["read_only"] and not report["submitted"]
    assert files(tmp_path) == before


@pytest.mark.parametrize(
    "bad_file",
    [
        "workflow_state/state.json",
        "workflow_state/ledgers/phase_data.json",
        "parameters/config_session.json",
    ],
)
def test_runtime_diagnosis_isolates_bad_files_and_keeps_other_sections(tmp_path, bad_file):
    runtime = project(tmp_path)
    write(tmp_path / "workflow_state/state.json", {"tasks": []})
    write(tmp_path / "workflow_state/ledgers/phase_data.json", {"branches": {}, "structures": {}})
    path = tmp_path / bad_file
    path.write_text('{\n "value": }', encoding="utf-8")
    before = files(tmp_path)
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["status"] == "needs_attention"
    assert report["sections"]["runtime"]["available"]
    issue = next(row for row in report["issues"] if row.get("file") == str(path))
    assert issue["code"] == "invalid_json" and issue["line"] == 2
    assert issue["next_step"]
    assert report["sections"]["operations"]["available"]
    assert files(tmp_path) == before


def test_corrupt_runtime_reports_location_without_echoing_secrets(tmp_path):
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text('{\n"api_key":"DO_NOT_PRINT",\n"state_path":}', encoding="utf-8")
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["status"] == "needs_attention"
    assert report["issues"][0]["line"] == 3
    assert "DO_NOT_PRINT" not in json.dumps(report)


def test_missing_confirmed_state_is_not_mistaken_for_new_project(tmp_path):
    runtime = project(tmp_path)
    session_path = tmp_path / "parameters/config_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    session.update(
        status="confirmed", confirmed_snapshot={"config": session["config"], "config_version": "v1"}
    )
    write(session_path, session)
    write(
        tmp_path / "workflow_state/ledgers/phase_data.json",
        {"branches": {"b1": {}}, "structures": {}},
    )
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["status"] == "needs_attention"
    assert any(issue["code"] == "missing_file" for issue in report["issues"])


def test_doctor_cli_accepts_runtime_and_emits_json(tmp_path):
    import os
    import subprocess
    import sys

    runtime = project(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "phase_agent.runtime.project_diagnostics",
            "--runtime",
            str(runtime),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["read_only"]


def test_doctor_follows_confirmed_storage_and_reports_conflicting_old_state(tmp_path):
    from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage

    runtime = project(tmp_path)
    session_path = tmp_path / "parameters/config_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    storage = default_workspace_storage(tmp_path)
    storage["paths"].update(
        state="workflow_state/selected.json", ledger="workflow_state/selected-ledger.json"
    )
    session["config"]["storage"] = storage
    session.update(
        status="confirmed", confirmed_snapshot={"config": session["config"], "config_version": "v1"}
    )
    write(session_path, session)
    write(tmp_path / storage["paths"]["state"], {"tasks": [], "confirmed_config_version": "v1"})
    write(tmp_path / storage["paths"]["ledger"], {"branches": {}, "structures": {}})
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["sections"]["binding"]["data"]["valid"]
    assert report["sections"]["storage"]["data"]["state"].endswith("selected.json")
    assert report["sections"]["ledger"]["available"]
    write(tmp_path / "workflow_state/state.json", {"tasks": []})
    before = files(tmp_path)
    report = project_diagnostics.inspect_runtime(runtime)
    assert report["status"] == "needs_attention"
    assert not report["sections"]["storage"]["available"]
    assert files(tmp_path) == before


def test_doctor_distinguishes_historical_feedback_error_without_clearing_it(tmp_path):
    runtime = project(tmp_path)
    write(
        tmp_path / "workflow_state/state.json",
        {
            "tasks": [{"task_id": "mc", "status": "completed", "outputs": {"pool_summary": []}}],
            "scientific_feedback_errors": {
                "mc": {"code": "result_output_object_required:pool_summary"}
            },
            "feedback_processed_task_ids": ["mc"],
        },
    )
    write(tmp_path / "workflow_state/ledgers/phase_data.json", {})
    before = files(tmp_path)
    report = project_diagnostics.inspect_runtime(runtime)
    facts = report["sections"]["state"]["data"]
    assert facts["scientific_feedback_recheck"]["valid_now"] == 1
    assert facts["scientific_feedback_recheck"]["already_recorded"] == 1
    assert facts["scientific_feedback_errors"]["mc"]["code"].endswith("pool_summary")
    assert files(tmp_path) == before
