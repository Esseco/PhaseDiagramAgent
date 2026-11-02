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
    with patch("phase_agent.runtime.phase_diagram_chat.phase_diagram_reply", return_value="plot.png") as plot:
        assert control.codex_plot() == {"reply": "plot.png"}
    plot.assert_called_once_with({"active_model_version": "model-1"}, operation="plot", method="mlip")
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
    state.update(active_model_version="m2", phase_diagrams={"mlip": fresh},
                 phase_diagrams_by_model={"m1": old, "m2": fresh},
                 analysis_output_directory=str(tmp_path / "analysis_outputs"))
    path = tmp_path / "state.json"
    original = json.dumps(state)
    path.write_text(original, encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(json.dumps({"state_path": str(path), "phase_diagram_directory": state["analysis_output_directory"], "local_path_relocations": [{"from": "must-not-migrate", "to": "unused"}]}), encoding="utf-8")
    server = CodexMcpServer(runtime_config=runtime, analysis_only=True)
    denied = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "phase_propose", "arguments": {"action": {"tool": "generate_branches"}}}})
    assert "analysis_only" in denied["error"]["message"]
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "phase_plot", "arguments": {"refresh_analysis": True}}})
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
