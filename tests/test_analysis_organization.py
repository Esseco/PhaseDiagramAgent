import json
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from phase_agent.analysis.feedback.analysis_output_organization import organize_analysis_outputs
from phase_agent.analysis.feedback.versioned_products import publish_bundle
from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory


def fixture(project):
    root = project / "analysis_outputs"
    root.mkdir()
    frame = project / "frame.vasp"
    frame.write_text("exact frame", encoding="utf8")
    digest = sha256(frame.read_bytes()).hexdigest()
    entries = [
        {"composition": {"O": 2}, "eform_per_O2": 0, "phase": "O1", "is_stable": True},
        {"composition": {"Na": 0.5, "O": 2}, "eform_per_O2": -0.1, "phase": "P3", "is_stable": True, "structure_sha256": digest},
        {"composition": {"Na": 1, "O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": True},
    ]
    snapshot = {"status": "completed", "model_version": "m", "version": "v", "entries": entries}
    def build(folder):
        Image.new("RGB", (8, 8), "white").save(folder / "main.png")
        Image.new("RGB", (8, 8), "white").save(folder / "hd_cross.png")
        (folder / "phase_diagram.json").write_text(json.dumps(snapshot), encoding="utf8")
        (folder / "report.md").write_text("report", encoding="utf8")
    publish_bundle(root, kind="phase_diagram", epoch="epoch0", model="m", identity={"hull_version": "v"},
                   required=("main.png", "hd_cross.png", "phase_diagram.json", "report.md"), build=build)
    decision = {"candidate_id": "S", "action": "DFT_SINGLE_POINT", "structure_path": str(frame), "phase_diagram_version": "v"}
    plan = project / "plan.json"
    plan.write_text(json.dumps({"model_version": "m", "hull_version": "v", "decisions": [decision],
                                "actual": {"tasks": [{"structure_id": "S", "structure_path": str(frame)}]}}), encoding="utf8")
    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 1}},
             "qbc_selection_history": [{"artifacts": {"json": str(plan)}}]}
    return root, frame, plan, state


def test_organization_preserves_explicit_failed_epoch_archive(tmp_path):
    root, _, _, state = fixture(tmp_path)
    archive = root / "epoch1-extrapolation_failed"
    archive.mkdir()
    (archive / "EXTRAPOLATION_ARCHIVE.md").write_text("retained", encoding="utf8")
    organize_analysis_outputs(state, root)
    assert (archive / "EXTRAPOLATION_ARCHIVE.md").read_text(encoding="utf8") == "retained"
    assert archive.parent == root


def test_agent_organization_migrates_and_plots_exact_frame(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    (root / "epoch0_old_report.md").write_text("retained", encoding="utf8")
    (root / "round_summary.csv").write_text("summary", encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed", result
    assert {p.name for p in root.iterdir()} == {"epoch0", "latest", "README.md", "history"}
    assert (root / "epoch0/Historical_Reports/epoch0_old_report.md").read_text() == "retained"
    assert (analysis_metadata_directory(root) / "round_summary.csv").read_text() == "summary"
    assert (analysis_metadata_directory(root) / "output_index.md").is_file()
    assert (root / "epoch0/DFT_Selection/DFT_Selection_01.png").is_file()
    assert (root / "epoch0/phase_diagram/1_dft-selection-1/selection_01.png").is_file()
    assert result["dft_selections"]["epoch0"][0]["numbered_step"] == "1_dft-selection-1"
    assert not (root / "latest/DFT_Selection.png").exists()
    assert "Main phase diagram" in (root / "epoch0/README.md").read_text(encoding="utf8")
    second = organize_analysis_outputs(state, root)
    assert second["status"] == "completed"
    assert second["dft_selections"]["epoch0"][0]["reused"] is True
    assert {p.name for p in root.iterdir()} == {"epoch0", "latest", "README.md", "history"}


def test_incorrect_geometry_never_publishes_selection(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    frame.write_text("changed frame", encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "incomplete"
    assert not (root / "epoch0/DFT_Selection/DFT_Selection_01.png").exists()
    assert "unavailable" in (root / "epoch0/README.md").read_text(encoding="utf8")
    assert not (root / "latest/DFT_Selection.png").exists()


def test_missing_saved_hull_never_uses_a_different_hull(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    data = json.loads(plan.read_text(encoding="utf8"))
    data["hull_version"] = "old"
    data["decisions"][0]["phase_diagram_version"] = "old"
    plan.write_text(json.dumps(data), encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "incomplete"
    assert result["dft_selections"]["epoch0"][0]["reason"] == "saved_hull_version_unavailable"
    assert not (root / "epoch0/DFT_Selection/DFT_Selection_01.png").exists()


def test_organization_runs_through_existing_agent_mcp_without_state_changes(tmp_path, monkeypatch):
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    root, frame, plan, state = fixture(tmp_path)
    state["analysis_output_directory"] = str(root)
    state_path = tmp_path / "state.json"
    original = json.dumps(state)
    state_path.write_text(original, encoding="utf8")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": str(state_path), "phase_diagram_directory": str(root)}), encoding="utf8")
    monkeypatch.setattr("phase_agent.analysis.feedback.export_dft_products.publish_output_catalog", lambda state, directory: None)
    server = CodexMcpServer(runtime_config=config, analysis_only=True)
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "phase_plot", "arguments": {"refresh_analysis": True, "organize_analysis": True}}})
    assert "error" not in response, response
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["organization"]["status"] == "completed"
    assert (root / "epoch0/DFT_Selection/DFT_Selection_01.png").is_file()
    assert state_path.read_text(encoding="utf8") == original


def test_agent_updates_only_relocated_navigation_paths(tmp_path, monkeypatch):
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    root, frame, plan, state = fixture(tmp_path)
    state["analysis_output_directory"] = str(root)
    state["output_index_path"] = str(root / "output_index.md")
    state["round_summary_csv_path"] = str(root / "round_summary.csv")
    (root / "output_index.md").write_text("old index", encoding="utf8")
    (root / "round_summary.csv").write_text("summary", encoding="utf8")
    (root / "latest/output_index.md").write_text("new index", encoding="utf8")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf8")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": str(state_path), "phase_diagram_directory": str(root)}), encoding="utf8")
    monkeypatch.setattr("phase_agent.analysis.feedback.export_dft_products.publish_output_catalog", lambda state, directory: None)
    server = CodexMcpServer(runtime_config=config, analysis_only=True)
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "phase_plot", "arguments": {"refresh_analysis": True, "organize_analysis": True}}})
    assert "error" not in response, response
    saved = json.loads(state_path.read_text(encoding="utf8"))
    assert saved["output_index_path"] == str(analysis_metadata_directory(root) / "output_index.md")
    assert saved["round_summary_csv_path"] == str(analysis_metadata_directory(root) / "round_summary.csv")
    assert {k:v for k,v in saved.items() if k not in {"output_index_path", "round_summary_csv_path"}} == {k:v for k,v in state.items() if k not in {"output_index_path", "round_summary_csv_path"}}


def test_frame_qualified_selection_requires_matching_structure_hash(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    data = json.loads(plan.read_text(encoding="utf8"))
    digest = sha256(frame.read_bytes()).hexdigest()
    data["decisions"][0]["candidate_id"] = "S:frame:" + digest
    plan.write_text(json.dumps(data), encoding="utf8")
    assert organize_analysis_outputs(state, root)["status"] == "completed"
    data["decisions"][0]["candidate_id"] = "S:frame:" + "0" * 64
    plan.write_text(json.dumps(data), encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "incomplete"
    assert result["dft_selections"]["epoch0"][0]["reason"] == "selected_structure_hash_changed"
    assert not (root / "epoch0/DFT_Selection/DFT_Selection_01.png").exists()


def test_organize_only_does_not_repeat_phase_analysis(tmp_path, monkeypatch):
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    root, frame, plan, state = fixture(tmp_path)
    state["analysis_output_directory"] = str(root)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf8")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": str(state_path), "phase_diagram_directory": str(root)}), encoding="utf8")
    def forbidden(*args, **kwargs):
        raise AssertionError("must not repeat scientific analysis")
    monkeypatch.setattr("phase_agent.analysis.feedback.export_dft_products.publish_output_catalog", forbidden)
    server = CodexMcpServer(runtime_config=config, analysis_only=True)
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "phase_plot", "arguments": {"organize_analysis": True}}})
    assert "error" not in response, response
    assert json.loads(response["result"]["content"][0]["text"])["organization"]["status"] == "completed"


def test_windows_extended_paths_are_recognized_inside_project(tmp_path):
    import os
    if os.name != "nt":
        pytest.skip("Windows extended paths")
    from phase_agent.tools.remote.transfer_bundle import local_path
    root, frame, plan, state = fixture(tmp_path)
    data = json.loads(plan.read_text(encoding="utf8"))
    extended = str(local_path(frame))
    data["decisions"][0]["structure_path"] = extended
    data["actual"]["tasks"][0]["structure_path"] = extended
    plan.write_text(json.dumps(data), encoding="utf8")
    assert organize_analysis_outputs(state, root)["status"] == "completed"


def test_saved_hull_path_relocation_requires_matching_frame_hash_and_science(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    original = json.loads((root / "epoch0/phase_diagram/latest/phase_diagram.json").read_text(encoding="utf8"))
    old = json.loads(json.dumps(original))
    old["entries"][1]["structure_path"] = "old-location.vasp"
    path = root / "epoch0/phase_diagrams/mlip/history/phase_diagram_mlip_v.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old), encoding="utf8")
    first = organize_analysis_outputs(state, root)
    assert first["status"] == "completed" and first["dft_selections"]["epoch0"][0]["status"] == "completed"
    old["entries"][1]["eform_per_O2"] = -9
    path.write_text(json.dumps(old), encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "incomplete"
    assert result["dft_selections"]["epoch0"][0]["reason"] == "conflicting_saved_hull_versions"


def test_not_ready_reason_is_retained_without_bundle(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import write_epoch_guide
    epoch = tmp_path / "epoch0"
    (epoch / "phase_diagram").mkdir(parents=True)
    (epoch / "phase_diagram/latest_status.json").write_text(json.dumps({"status": "not_ready", "reason": "incomplete_import"}), encoding="utf8")
    missing = write_epoch_guide(epoch)
    assert missing["reason"] == "incomplete_import"
    assert "incomplete_import" in (epoch / "README.md").read_text(encoding="utf8")


def test_recorded_verified_layout_relocation_finds_exact_frame(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    old = tmp_path / "submissions/epoch0/old/frame.vasp"
    new = tmp_path / "submissions/epoch0/new/frame.vasp"
    new.parent.mkdir(parents=True)
    new.write_bytes(frame.read_bytes())
    data = json.loads(plan.read_text(encoding="utf8"))
    data["decisions"][0]["structure_path"] = str(old)
    data["actual"]["tasks"][0]["structure_path"] = str(old)
    plan.write_text(json.dumps(data), encoding="utf8")
    receipt = tmp_path / "submissions/.layout-migration.json"
    receipt.write_text(json.dumps({"project": tmp_path.name, "status": "migrated", "immutable_files_verified": 1,
                                   "moves": [{"from": str(old.parent), "to": str(new.parent)}]}), encoding="utf8")
    assert organize_analysis_outputs(state, root)["status"] == "completed"
    new.write_text("modified", encoding="utf8")
    assert organize_analysis_outputs(state, root)["status"] == "incomplete"


def test_mixed_historical_hulls_are_plotted_separately(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    frame2 = tmp_path / "other.vasp"
    frame2.write_text("different frame", encoding="utf8")
    old = json.loads((root / "epoch0/phase_diagram/latest/phase_diagram.json").read_text(encoding="utf8"))
    old["version"] = "v2"
    old["entries"][1]["structure_sha256"] = sha256(frame2.read_bytes()).hexdigest()
    history = root / "epoch0/phase_diagrams/mlip/history/phase_diagram_mlip_v2.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps(old), encoding="utf8")
    data = json.loads(plan.read_text(encoding="utf8"))
    data.pop("hull_version")
    data["decisions"].append({"candidate_id": "T", "action": "DFT_SINGLE_POINT", "structure_path": str(frame2), "phase_diagram_version": "v2"})
    data["actual"]["tasks"].append({"structure_id": "T", "structure_path": str(frame2)})
    plan.write_text(json.dumps(data), encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed" and result["physical_layout"]["compatibility_aliases"], result
    assert {row["hull_version"] for row in result["dft_selections"]["epoch0"]} == {"v", "v2"}
    assert len(list((root / "epoch0/DFT_Selection").glob("*.png"))) == 2


def test_cancelled_selection_is_not_exposed_as_current(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    data = json.loads(plan.read_text(encoding="utf8"))
    data["actual"]["tasks"][0]["task_id"] = "cancelled"
    plan.write_text(json.dumps(data), encoding="utf8")
    state["tasks"] = [{"task_id": "cancelled", "status": "cancelled"}]
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed"
    assert result["discarded_selections"][0]["reason"] == "all_prepared_tasks_cancelled"
    assert not list((root / "epoch0/DFT_Selection").glob("*.png"))
    assert not (root / "latest/DFT_Selection.png").exists()


def test_missing_plan_reconstructed_from_unique_registered_same_frame_tasks(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    state["qbc_selection_history"] = [{"task_key": "round1", "accepted": [{"task_key": "K", "candidate_id": "S", "action": "DFT_SINGLE_POINT"}]}]
    state["tasks"] = [{"task_key": "K", "task_id": "T", "structure_id": "S", "structure_path": str(frame),
                       "model_version": "m", "stage": "dft_single_point", "status": "completed"}]
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed", result
    row = result["dft_selections"]["epoch0"][0]
    assert row["display_basis"] == "reconstructed_on_saved_epoch_hull_selection_time_hull_unrecorded"
    assert "selection time was not recorded" in (root / "epoch0/README.md").read_text(encoding="utf8")
    state["tasks"].append(dict(state["tasks"][0]))
    second = organize_analysis_outputs(state, root)
    assert second["status"] == "incomplete"
    assert second["dft_selections"]["unassigned"][0]["reason"] == "selection_task_identity_ambiguous"


def test_new_prepared_selection_updates_readable_view_without_full_reanalysis(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import publish_prepared_selection
    root, frame, plan, state = fixture(tmp_path)
    state["analysis_output_directory"] = str(root)
    assert organize_analysis_outputs(state, root)["status"] == "completed"
    record = json.loads(plan.read_text(encoding="utf8"))
    new_plan = tmp_path / "new-plan.json"
    new_plan.write_text(json.dumps(record), encoding="utf8")
    result = publish_prepared_selection(state, record, new_plan)
    assert result["status"] == "completed"
    assert Path(result["plots"][0]).is_file()
    assert result["analysis_step"]["directory"] == "2_dft-selection-2"
    assert "DFT selection 02" in (root / "epoch0/README.md").read_text(encoding="utf8")
    assert not (root / "latest/DFT_Selection.png").exists()
    assert publish_prepared_selection(state, record, new_plan)["status"] == "already_registered"


def test_latest_selection_is_not_shown_after_publication_becomes_unready(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import write_latest_guide
    root, frame, plan, state = fixture(tmp_path)
    result = organize_analysis_outputs(state, root)
    assert not (root / "latest/DFT_Selection.png").exists()
    (root / "latest/phase_diagram_status.json").write_text(json.dumps({"status": "not_ready", "reason": "new data pending"}), encoding="utf8")
    write_latest_guide(root, state, result["dft_selections"])
    assert not (root / "latest/DFT_Selection.png").exists()
    assert "new data pending" in (root / "latest/README.md").read_text(encoding="utf8")


def test_organization_keeps_root_evidence_referenced_by_saved_state(tmp_path):
    root, frame, plan, state = fixture(tmp_path)
    evidence = root / "endpoint_supplements" / "old" / "frame.vasp"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("registered", encoding="utf8")
    state["endpoint_supplement"] = {"directory": str(evidence.parent)}
    state["tasks"] = [{"outputs": {"structure_path": str(evidence)}}]
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "incomplete"
    assert result["protected_root_entries"][0]["reason"] == "referenced_by_saved_state"
    assert evidence.read_text(encoding="utf8") == "registered"
    assert not (root / "latest/Historical_Reports/endpoint_supplements").exists()


def test_latest_dft_error_requires_active_model_and_verified_status(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import write_latest_guide

    root, _, _, state = fixture(tmp_path)

    def build(folder):
        for name in ("energy_parity.png", "force_parity.png"):
            Image.new("RGB", (8, 8), "white").save(folder / name)
        (folder / "metrics.json").write_text("{}", encoding="utf8")

    publish_bundle(root, kind="dft_error", epoch="epoch0", model="m", identity={},
                   required=("energy_parity.png", "force_parity.png", "metrics.json"), build=build)
    write_latest_guide(root, state, {})
    guide = (root / "latest/README.md").read_text(encoding="utf8")
    assert "[Energy parity](dft_error/energy_parity.png)" in guide
    assert "[Force parity](dft_error/force_parity.png)" in guide
    state["active_model_version"] = "new-model"
    write_latest_guide(root, state, {})
    guide = (root / "latest/README.md").read_text(encoding="utf8")
    assert "Current-model DFT error is not ready" in guide
    assert "[Energy parity]" not in guide


def test_legacy_phase_directory_is_reported_until_safely_migrated(tmp_path):
    root, _, _, state = fixture(tmp_path)
    legacy = root / "epoch0/phase_diagrams/mlip/history"
    legacy.mkdir(parents=True)
    (legacy / "registered.json").write_text("{}", encoding="utf8")
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed"
    assert result["legacy_phase_directories"] == []
    assert (root / "history/epoch0/phase_diagrams/mlip/history/registered.json").read_text(encoding="utf8") == "{}"
    assert (legacy / "registered.json").is_file()


def test_old_model_error_bundle_is_archived_outside_latest(tmp_path):
    root, _, _, state = fixture(tmp_path)
    def build(folder):
        for name in ("energy_parity.png", "force_parity.png"):
            Image.new("RGB", (8, 8), "white").save(folder / name)
        (folder / "metrics.json").write_text("{}", encoding="utf8")
    publish_bundle(root, kind="dft_error", epoch="epoch0", model="old", identity={},
                   required=("energy_parity.png", "force_parity.png", "metrics.json"), build=build)
    result = organize_analysis_outputs(state, root)
    assert result["status"] == "completed"
    assert not (root / "latest/dft_error").exists()
    assert (analysis_metadata_directory(root) / "Previous_Latest/stale_dft_error/energy_parity.png").is_file()
    assert "Current-model DFT error is not ready" in (root / "latest/README.md").read_text(encoding="utf8")


def test_new_project_selection_publishes_without_prior_organization(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import publish_prepared_selection
    root, _, plan, state = fixture(tmp_path)
    state["analysis_output_directory"] = str(root)
    result = publish_prepared_selection(state, json.loads(plan.read_text(encoding="utf8")), plan)
    assert result["status"] == "completed"
    assert result["analysis_step"]["directory"] == "1_dft-selection-1"
    assert (root / "epoch0/phase_diagram/1_dft-selection-1/selection_01.png").is_file()
    again = organize_analysis_outputs(state, root)
    assert again["status"] == "completed"
    assert Path(again["dft_selections"]["epoch0"][0]["path"]).is_relative_to(root / "epoch0/phase_diagram/_internal/selection")
    assert not (root / "epoch0/DFT_Selection").exists()


def test_later_latest_cleanup_uses_relocated_history(tmp_path):
    root, _, _, state = fixture(tmp_path)
    previous = analysis_metadata_directory(root) / "Previous_Latest"
    previous.mkdir(parents=True)
    (previous / "old.txt").write_text("old evidence", encoding="utf8")
    organize_analysis_outputs(state, root)
    (root / "latest" / "new_diagnostic.txt").write_text("new evidence", encoding="utf8")
    organize_analysis_outputs(state, root)
    assert (root / "history/project/Previous_Latest/new_diagnostic.txt").read_text(encoding="utf8") == "new evidence"
    assert (previous / "old.txt").read_text(encoding="utf8") == "old evidence"
