from copy import deepcopy
import pytest

from phase_agent.decisions.agent.dialogue_contract import dialogue_errors, dialogue_output_contract
from phase_agent.decisions.agent.proposal_validation import proposal_errors
from phase_agent.runtime.phase_diagram_chat import phase_diagram_reply
from phase_agent.tools.step_runner.file_protocol import write_json, read_json
from tests.test_unified_react_dialogue import make_handler


def diagram():
    return {
        "active_model_version": "m1",
        "phase_diagrams": {
            "mlip": {
                "status": "completed", "model_version": "m1", "version": "hull-one",
                "entries": [
                    {"composition": {"Na": x, "O": 2}, "eform_per_O2": energy,
                     "ehull": 0, "phase": "O3", "phase_identification_status": "identified",
                     "is_stable": True}
                    for x, energy in [(0, 0), (0.5, -0.2), (1, 0)]
                ],
            }
        },
    }


def test_contract_and_no_scientific_context():
    action = {"kind": "phase_diagram", "operation": "plot"}
    assert not dialogue_errors(action, True)
    assert dialogue_errors({**action, "operation": "delete"}, True)
    assert dialogue_errors({**action, "method": "../../other"}, True)
    assert dialogue_errors({**action, "tool": "run_calculation_stage"}, True)
    assert proposal_errors(action, {"unified_dialogue": True, "decision_context": {}})
    assert "phase_diagram_schema" in dialogue_output_contract(allow_controls=True)
    assert "phase_diagram_schema" not in dialogue_output_contract()


def test_locate_does_not_write_or_mutate_state(tmp_path):
    state = diagram()
    before = deepcopy(state)
    root = tmp_path / "outputs"
    reply = phase_diagram_reply(state, operation="locate", directory=root)
    assert str(root.resolve()) in reply
    assert "尚未生成" in reply
    assert not root.exists()
    assert state == before


def test_plot_creates_real_artifacts_and_locate_reports_folder(tmp_path):
    state = diagram()
    before = deepcopy(state)
    reply = phase_diagram_reply(state, operation="plot", directory=tmp_path)
    assert "已绘制" in reply
    png, = tmp_path.rglob("*.png")
    assert not list(tmp_path.rglob("*.pdf"))
    assert not list(tmp_path.rglob("*.svg"))
    assert png.read_bytes().startswith(b"\x89PNG")
    from PIL import Image
    with Image.open(png) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=.02)
    for path in (png,):
        assert str(path.resolve()) in reply
    csv, = [p for p in tmp_path.rglob("phase_diagram.csv")]
    state["phase_diagrams"]["mlip"]["csv_path"] = str(csv)
    located = phase_diagram_reply(state, operation="locate")
    assert str(png.parent) in located
    assert str(csv) in located
    state["phase_diagrams"]["mlip"].pop("csv_path")
    assert state == before


@pytest.mark.parametrize("failure", ["empty", "stale", "unidentified", "nan"])
def test_unavailable_data_never_claims_success(tmp_path, failure):
    state = diagram()
    snapshot = state["phase_diagrams"]["mlip"]
    if failure == "empty":
        snapshot["entries"] = []
    elif failure == "stale":
        state["active_model_version"] = "m2"
    elif failure == "unidentified":
        snapshot["entries"][0]["phase_identification_status"] = "pending"
    else:
        snapshot["entries"][0]["eform_per_O2"] = float("nan")
    reply = phase_diagram_reply(state, operation="plot", directory=tmp_path)
    assert "未绘制" in reply and "已绘制" not in reply
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.parametrize(("message", "operation"), [
    ("帮我绘制现在的相图", "plot"), ("现在的相图文件夹在哪", "locate"),
    ("能给我画一张当前的相图吗", "plot"), ("刚才那个图保存到哪个目录了", "locate"),
])
def test_dialogue_model_dispatch_without_scientific_action(tmp_path, message, operation):
    calls = []
    def model(payload):
        calls.append(payload)
        return {"kind": "phase_diagram", "operation": operation}

    handler = make_handler(tmp_path, model)
    handler.workflow_kwargs["phase_diagram_directory"] = str(tmp_path / "outputs")
    state = diagram()
    state["pending_execution_policies"] = {}
    write_json(handler.state_path, state)
    reply = handler([{"role": "user", "content": message}], conversation_id="plot-chat")
    assert len(calls) == 1
    assert "phase_diagram_schema" in str(calls[0])
    assert ("已绘制" if operation == "plot" else "相图文件夹") in reply
    saved = read_json(handler.state_path, {})
    assert saved["pending_execution_policies"] == {}
    assert not saved.get("tasks")
    assert not saved.get("budget_reservations")
    assert not (tmp_path / "execution_receipts.sqlite").exists()


def test_dft_and_combined_do_not_use_mlip(tmp_path):
    state = diagram()
    for method in ("dft", "combined"):
        reply = phase_diagram_reply(state, operation="plot", method=method, directory=tmp_path)
        assert "未绘制" in reply
        assert not list(tmp_path.rglob("*.png"))


def test_old_diagram_location_is_returned_with_version_warning(tmp_path):
    state = diagram()
    state["active_model_version"] = "m2"
    csv = tmp_path / "old-model" / "phase_diagram.csv"
    state["phase_diagrams"]["mlip"]["csv_path"] = str(csv)
    reply = phase_diagram_reply(state, operation="locate")
    assert str(csv.parent) in reply
    assert "版本不一致" in reply
    assert not csv.parent.exists()


@pytest.mark.parametrize("method", ["dft", "combined"])
def test_explicit_methods_plot_the_selected_saved_snapshot(tmp_path, method):
    state = diagram()
    if method == "combined":
        phase_diagram_reply(state, operation="plot", directory=tmp_path)
        csv, = tmp_path.rglob("phase_diagram.csv")
        state["phase_diagrams"]["mlip"]["csv_path"] = str(csv)
    state["phase_diagrams"][method] = state["phase_diagrams"].pop("mlip")
    reply = phase_diagram_reply(state, operation="plot", method=method, directory=tmp_path)
    assert f"已绘制当前 {method.upper()}" in reply


def test_local_plotter_receives_phase_hulls_and_mev(tmp_path, monkeypatch):
    import Process_Phasediagram
    calls = []
    original = Process_Phasediagram.plot_phase_diagram

    def record(x, energy, phases, stable, **kwargs):
        result = original(x, energy, phases, stable, **kwargs)
        calls.append((x, energy, phases, stable, kwargs, result))
        return result

    monkeypatch.setattr(Process_Phasediagram, "plot_phase_diagram", record)
    state = diagram()
    state["upload_layout"] = {"model_rounds": {"m1": 2}}
    before = deepcopy(state)
    reply = phase_diagram_reply(state, operation="plot", directory=tmp_path)
    assert "分相已开启" in reply and "epoch1" in reply
    assert len(calls) == 1
    x, energy, phases, stable, options, (_, axes) = calls[0]
    assert x == [0, 0.5, 1]
    assert energy == [0, -200, 0]
    assert phases == ["O3"] * 3 and stable == [True] * 3
    assert options["phase_hulls"] is True
    assert axes.get_ylabel() == "Formation energy (meV/O2)"
    assert any(line.get_linestyle() == "--" for line in axes.lines)
    assert state == before


def test_explicit_vector_formats_are_forwarded_by_dialogue(tmp_path):
    action = {"kind": "phase_diagram", "operation": "plot", "formats": ["png", "pdf", "svg"]}
    assert not dialogue_errors(action, True)
    assert dialogue_errors({**action, "formats": []}, True)
    assert dialogue_errors({**action, "formats": ["../../bad"]}, True)
    handler = make_handler(tmp_path, lambda payload: action)
    output = tmp_path / "outputs"
    handler.workflow_kwargs["phase_diagram_directory"] = str(output)
    state = diagram()
    write_json(handler.state_path, state)
    reply = handler([{"role": "user", "content": "绘制当前相图，同时导出PDF和SVG"}], conversation_id="vector-chat")
    assert "已绘制" in reply
    assert {p.suffix for p in output.rglob("*") if p.suffix in {".png", ".pdf", ".svg"}} == {".png", ".pdf", ".svg"}
    assert next(output.rglob("*.pdf")).read_bytes().startswith(b"%PDF")
    assert "Formation energy" in next(output.rglob("*.svg")).read_text(encoding="utf-8")
    assert not read_json(handler.state_path, {}).get("tasks")
