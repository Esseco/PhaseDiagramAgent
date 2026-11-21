from phase_agent.analysis.phase import phase_quality
from phase_agent.analysis.phase.phase_quality_reports import quality_report_read_scope
from phase_agent.runtime.codex import context


def test_scoped_geometry_reuse_requires_same_rows_tasks_and_files(tmp_path, monkeypatch):
    final = tmp_path / "final.vasp"
    initial = tmp_path / "initial.vasp"
    final.write_text("final", encoding="utf8")
    initial.write_text("initial", encoding="utf8")
    state = {
        "active_model_version": "m",
        "tasks": [
            {"task_id": "t", "model_version": "m", "structure_path": str(initial), "phase": "O3"}
        ],
    }
    point = {"key": "g", "rows": [{"structure_path": str(final), "source_task_id": "t"}]}
    calls = []
    monkeypatch.setattr(context, "shared_codex_identity", lambda: {"version": "v"})
    monkeypatch.setattr(
        phase_quality, "_geometry_evidence", lambda p, s: calls.append(1) or {"count": len(calls)}
    )
    with quality_report_read_scope():
        assert phase_quality._cached_geometry(point, state, {})["count"] == 1
        assert phase_quality._cached_geometry(point, state, {})["count"] == 1
        final.write_text("changed size", encoding="utf8")
        assert phase_quality._cached_geometry(point, state, {})["count"] == 2
        state["tasks"][0]["phase"] = "P3"
        assert phase_quality._cached_geometry(point, state, {})["count"] == 3
    with quality_report_read_scope():
        assert phase_quality._cached_geometry(point, state, {})["count"] == 4
