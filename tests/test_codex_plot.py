import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from phase_agent.runtime.local_agent_control import LocalAgentControl


def test_codex_plot_uses_existing_plotter_without_changing_state(tmp_path):
    path = tmp_path / "state.json"
    original = json.dumps({"active_model_version": "model-1"})
    path.write_text(original, encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with patch(
        "phase_agent.runtime.phase_diagram_chat.phase_diagram_reply", return_value="plot.png"
    ) as plot:
        assert control.codex_plot() == {"reply": "plot.png"}
    plot.assert_called_once_with(
        {"active_model_version": "model-1"}, operation="plot", method="mlip", directory=None
    )
    assert path.read_text(encoding="utf-8") == original


def test_analysis_refresh_backfills_history_through_public_control_without_state_writes(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from tests.test_analysis_publication import phase_state
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    state = phase_state()
    old = deepcopy(state["phase_diagrams"]["mlip"])
    fresh = deepcopy(old)
    fresh.update(model_version="m2", version="new-model-hull")
    for row in fresh["entries"]:
        row["source_version"] = "m2"
    state.update(
        active_model_version="m2",
        phase_diagrams={"mlip": fresh},
        phase_diagrams_by_model={"m1": old, "m2": fresh},
        analysis_output_directory=str(tmp_path / "analysis_outputs"),
    )
    path = tmp_path / "state.json"
    original = json.dumps(state)
    path.write_text(original, encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "state_path": str(path),
                "phase_diagram_directory": state["analysis_output_directory"],
                "local_path_relocations": [{"from": "must-not-migrate", "to": "unused"}],
            }
        ),
        encoding="utf-8",
    )
    server = CodexMcpServer(runtime_config=runtime, analysis_only=True)
    denied = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "phase_propose",
                "arguments": {"action": {"tool": "generate_branches"}},
            },
        }
    )
    assert "analysis_only" in denied["error"]["message"]
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "phase_plot", "arguments": {"refresh_analysis": True}},
        }
    )
    assert "error" not in response, response
    result = json.loads(response["result"]["content"][0]["text"])
    assert result["analysis_publications"]["phase_diagram"]["status"] == "completed"
    assert result["phase_epoch_publications"]["m1"]["status"] == "completed", result
    root = tmp_path / "analysis_outputs"
    current = validate_bundle(root / "latest" / "phase_diagram")
    history = validate_bundle(root / "epoch0" / "phase_diagram" / "latest")
    assert current["identity"]["model_version"] == "m2"
    assert history["identity"]["model_version"] == "m1"
    assert history["identity"]["publication_basis"] == "saved_historical_hull_current_review_rules"
    for epoch in ("epoch0", "epoch1"):
        for name in ("main.png", "hd_cross.png"):
            assert (root / epoch / "phase_diagram" / "latest" / name).is_file()
    archive = Path(result["phase_epoch_publications"]["m1"]["manifest_path"])
    stamp = archive.stat().st_mtime_ns
    second = control.codex_plot(refresh_analysis=True)
    assert second["phase_epoch_publications"]["m1"] == result["phase_epoch_publications"]["m1"]
    assert archive.stat().st_mtime_ns == stamp
    assert path.read_text(encoding="utf-8") == original


def test_refresh_option_rejects_missing_registered_output_and_invalid_type(tmp_path):
    import pytest

    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="registered analysis"):
        control.codex_plot(refresh_analysis=True)
    with pytest.raises(ValueError, match="boolean"):
        control.codex_plot(refresh_analysis="true")
    assert path.read_text(encoding="utf-8") == "{}"


def test_analysis_only_binds_unregistered_legacy_standard_paths_without_state_write(
    tmp_path, monkeypatch
):
    from phase_agent.runtime.saved_analysis_project import SavedAnalysisProject

    root = tmp_path
    state_path = root / "workflow_state" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text("{}", encoding="utf8")
    config = root / "agent_runtime.json"
    config.write_text(
        json.dumps(
            {
                "state_path": "workflow_state/state.json",
                "phase_diagram_directory": "analysis_outputs",
            }
        ),
        encoding="utf8",
    )
    called = []
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.export_dft_products.publish_output_catalog",
        lambda state, directory: called.append(str(directory)),
    )
    project = SavedAnalysisProject(config)
    project.control.codex_plot(refresh_analysis=True)
    assert called == [str(root / "analysis_outputs")]
    assert state_path.read_text(encoding="utf8") == "{}"
    config.write_text(
        json.dumps({"state_path": "workflow_state/state.json", "phase_diagram_directory": "other"}),
        encoding="utf8",
    )
    import pytest

    with pytest.raises(ValueError, match="standard project-local paths"):
        SavedAnalysisProject(config)


def test_plot_reads_only_version_bound_plot_fields(tmp_path, monkeypatch):
    from phase_agent.persistence.state.field_index import open_indexed_state
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.runtime.plot_state_projection import read_plot_state

    path = tmp_path / "state.json"
    state = {
        "active_model_version": "m",
        "phase_diagrams": {"mlip": {"version": "v"}},
        "tasks": [{"task_id": "a"}],
        "phase_records": [{"record_id": "r"}],
        "event_history": [{"irrelevant": "x" * 1000}],
        "decisions": [{"irrelevant": "y"}],
    }
    write_json(path, state)
    assert open_indexed_state(path) is not None
    monkeypatch.setattr(
        "phase_agent.tools.step_runner.file_protocol.read_json",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("full state read")),
    )
    projected = read_plot_state(path)
    assert projected == {
        k: state[k] for k in ("active_model_version", "phase_diagrams", "tasks", "phase_records")
    }
    assert json.loads(path.read_text(encoding="utf8")) == state


def test_plot_falls_back_to_authoritative_state_for_stale_index(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.runtime.plot_state_projection import read_plot_state

    path = tmp_path / "state.json"
    write_json(path, {"active_model_version": "m1", "event_history": ["old"]})
    replacement = {"active_model_version": "m2", "event_history": ["new"]}
    path.write_text(json.dumps(replacement), encoding="utf8")
    assert read_plot_state(path) == replacement


def test_plot_uses_current_in_memory_state_instead_of_stale_index(tmp_path, monkeypatch):
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.runtime.plot_state_projection import read_plot_state

    path = tmp_path / "state.json"
    write_json(path, {"active_model_version": "disk"})
    current = {"active_model_version": "memory"}
    monkeypatch.setattr(
        "phase_agent.runtime.turn_process.has_reusable_initial_state", lambda _: True
    )
    monkeypatch.setattr(
        "phase_agent.tools.step_runner.file_protocol.read_json", lambda *a, **k: current
    )
    assert read_plot_state(path) == current


def test_control_plot_uses_index_without_loading_unrelated_history(tmp_path, monkeypatch):
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    original = {
        "active_model_version": "m1",
        "phase_diagrams": {"mlip": {"version": "v"}},
        "tasks": [{"task_id": "a"}],
        "event_history": [{"large": "x" * 1000}],
    }
    write_json(path, original)
    observed = []
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    monkeypatch.setattr(
        "phase_agent.runtime.phase_diagram_chat.phase_diagram_reply",
        lambda state, **kw: observed.append(state) or "plotted",
    )
    monkeypatch.setattr(
        "phase_agent.tools.step_runner.file_protocol.read_json",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("full read")),
    )
    assert control.codex_plot() == {"reply": "plotted"}
    assert observed[0]["tasks"] == original["tasks"] and "event_history" not in observed[0]
    assert json.loads(path.read_text(encoding="utf8")) == original


def test_analysis_only_organizes_just_dft_summary(tmp_path):
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    output = tmp_path / "analysis_outputs"
    legacy = output / "epoch0/dft_error"
    legacy.mkdir(parents=True)
    (legacy / "latest_status.json").write_bytes(b"saved")
    other = output / "epoch0/DFT_Selection"
    other.mkdir()
    state = tmp_path / "workflow_state/state.json"
    state.parent.mkdir()
    state.write_bytes(b"{}")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": str(state), "phase_diagram_directory": str(output)}))
    server = CodexMcpServer(runtime_config=config, analysis_only=True)
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "phase_plot", "arguments": {"organize_dft": True}},
    })
    assert "error" not in response, response
    layout = json.loads(response["result"]["content"][0]["text"])["dft_layout"]
    assert layout["moved_directories"] == 1
    assert (output / "epoch0/dft/summary/latest_status.json").read_bytes() == b"saved"
    assert (legacy / "latest_status.json").read_bytes() == b"saved"
    assert other.is_dir() and not other.is_junction()
    assert state.read_bytes() == b"{}"
    guide = (output / "epoch0/README.md").read_text(encoding="utf-8")
    assert "dft/summary" in guide and "(<dft_error>)" not in guide


def test_round_error_entry_also_publishes_kfold_and_reports_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "analysis_outputs"
    root.mkdir()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"analysis_output_directory": str(root)}), encoding="utf8")
    control = LocalAgentControl(SimpleNamespace(state_path=state_path, lock=threading.RLock()))
    calls = []
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.round_error_trend.publish_round_error_trend",
        lambda state, directory: calls.append("round") or {"rows": []},
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.training_parity.publish_training_parity",
        lambda state, directory: calls.append("kfold") or {"latest": {"status": "unavailable"}},
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.analysis_output_organization.write_latest_guide",
        lambda *args: None,
    )
    report = control.codex_plot(round_error_trend=True)
    assert calls == ["round", "kfold"]
    assert report["training_parity"]["latest"]["status"] == "unavailable"
    assert json.loads(state_path.read_text(encoding="utf8")) == {"analysis_output_directory": str(root)}


def test_publish_phase_uses_indexed_projection_and_does_not_read_full_state(tmp_path, monkeypatch):
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    original = {
        "active_model_version": "m1", "active_model": {"sha256": "digest"},
        "model_refresh": {"hull_source_policy": "current_model_only_v1"},
        "analysis_output_directory": str(tmp_path / "analysis"),
        "phase_diagrams": {"mlip": {"version": "v"}}, "event_history": [{"large": "x" * 1000}],
    }
    write_json(path, original)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    observed = []
    monkeypatch.setattr("phase_agent.analysis.feedback.analysis_publication._phase",
                        lambda state, directory: observed.append(state) or {"status": "completed"})
    monkeypatch.setattr("phase_agent.tools.step_runner.file_protocol.read_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("full read")))
    assert control.codex_publish_phase() == {"status": "completed"}
    assert observed[0]["active_model"]["sha256"] == "digest"
    assert observed[0]["model_refresh"]["hull_source_policy"] == "current_model_only_v1"
    assert "event_history" not in observed[0]
    assert json.loads(path.read_text(encoding="utf8")) == original
