import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax
from phase_agent.tools.remote.finalize_vasp import finalize_vasp


def test_generated_wrapper_preserves_original_workflow_error(tmp_path, monkeypatch):
    source = tmp_path / "source.vasp"
    source.write_text("fake structure")
    folder = tmp_path / "task"

    def generator(directory, structure, **kwargs):
        directory.mkdir()
        (directory / "workflow.py").write_text("pass")

    prepare_pycode_relax(
        {"stage": "dft_single_point", "structure_path": str(source), "work_directory": str(folder)},
        manager=SimpleNamespace(data={}),
        generator=generator,
    )
    (folder / "task.json").write_text(json.dumps({"task_id": "t", "stage": "dft_single_point"}))

    def fail(*a, **k):
        raise ValueError("Static output needs WAVECAR for subsequent calculations")

    calls = []
    monkeypatch.setattr("runpy.run_path", fail)
    monkeypatch.setattr(
        "phase_agent.tools.remote.finalize_vasp.finalize_vasp",
        lambda *a, **k: calls.append(k) or {"status": "failed"},
    )
    with pytest.raises(ValueError, match="WAVECAR"):
        exec(
            compile((folder / "workflow.py").read_text(), "workflow.py", "exec"),
            {"__file__": str(folder / "workflow.py")},
        )
    assert calls[0]["exit_code"] == 1
    assert calls[0]["workflow_error"] == {
        "type": "ValueError",
        "message": "Static output needs WAVECAR for subsequent calculations",
    }


def test_finalizer_reports_workflow_failure_and_retains_labels(tmp_path, monkeypatch):
    task = {
        "task_id": "T",
        "task_key": "t",
        "batch_id": "b",
        "stage": "dft_single_point",
        "task_checksum": "original",
        "protocol_version": 1,
        "input_file_version": "v1",
    }
    (tmp_path / "task.json").write_text(json.dumps(task))
    parsed = {
        "status": "failed",
        "converged": False,
        "error": "vasp exited with code 1",
        "outputs": {"energy": -1.0, "xml_complete": True},
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.finalize_vasp.parse_vasp_result", lambda *a, **k: parsed
    )
    result = finalize_vasp(
        tmp_path,
        exit_code=1,
        calculation_directory=tmp_path,
        workflow_error={
            "type": "ValueError",
            "message": "Static output needs WAVECAR for subsequent calculations",
        },
    )
    assert result["status"] == "failed" and result["converged"] is False
    assert "workflow" in result["error"].lower() and "WAVECAR" in result["error"]
    assert "vasp exited with code 1" not in result["error"]
    assert result["outputs"]["energy"] == -1.0 and result["outputs"]["xml_complete"] is True
    assert result["workflow_exit_code"] == 1


@pytest.mark.parametrize(
    "entry",
    [
        "from Process_Vasp.workflows.atomate_runner import main",
        "from Process_Vasp.atomate_runner import main",
        "from atomate_runner import main",
    ],
)
def test_generated_task_carries_authoritative_runner(tmp_path, entry):
    from Process_Vasp.workflows import atomate_runner

    source = tmp_path / "source.vasp"
    source.write_text("fake geometry")
    target = tmp_path / "input"

    def generator(directory, structure, **kwargs):
        directory.mkdir()
        (directory / "workflow.py").write_text(entry + "\n")

    prepare_pycode_relax(
        {"stage": "dft_single_point", "structure_path": str(source), "work_directory": str(target)},
        manager=SimpleNamespace(data={}),
        generator=generator,
    )
    helper = target / "_handoff_atomate_runner.py"
    assert helper.read_bytes() == Path(atomate_runner.__file__).read_bytes()
    code = (target / "atomate_relax.py").read_text()
    assert "from _handoff_atomate_runner import main" in code and entry not in code
    compile(helper.read_text(encoding="utf-8"), "runner.py", "exec")
