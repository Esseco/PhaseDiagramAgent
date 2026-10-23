from copy import deepcopy
from pathlib import Path

import pytest

from phase_agent.analysis.phase.phase_round_analysis import publish_relax_mc_analysis
from phase_agent.analysis.phase.phase_quality_reports import quality_summary
from tests.test_phase_quality import sample


def registered(tmp_path):
    state = sample()
    snapshot = state["phase_diagrams"]["mlip"]
    snapshot["csv_path"] = str(tmp_path / "phase_diagram.csv")
    state["relax_mc_batches"] = {
        "b1": {
            "status": "completed",
            "model_version": "m1",
            "plan_checksum": "plan",
            "import_checksum": "receipt",
            "result_summary": {
                "status": "completed",
                "model_version": "m1",
                "plan_checksum": "plan",
            },
        }
    }
    return state


def test_complete_round_generates_once_and_reuses_derived_receipt(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as plotting

    calls = []
    image = tmp_path / "plot.png"
    image.write_bytes(b"real-file-stand-in")

    def plot(state, **kwargs):
        calls.append(kwargs)
        return f"已绘制当前 MLIP 相图：`{image}`"

    monkeypatch.setattr(plotting, "phase_diagram_reply", plot)
    state = registered(tmp_path)
    before = deepcopy(state)
    first = publish_relax_mc_analysis(state, "b1")
    second = publish_relax_mc_analysis(state, "b1")
    assert first["status"] == "completed" and first == second
    assert len(calls) == 1
    assert first["bundle_id"] == "b1" and first["hull_version"] == "h1"
    assert Path(first["receipt_path"]).is_file()
    assert state == before


@pytest.mark.parametrize(
    "change",
    ["partial", "failed", "stale", "unverified", "plan_mismatch", "old_summary", "task_wait"],
)
def test_incomplete_or_unbound_rounds_never_plot(tmp_path, monkeypatch, change):
    import phase_agent.runtime.phase_diagram_chat as plotting

    monkeypatch.setattr(
        plotting, "phase_diagram_reply", lambda *a, **kw: pytest.fail("must not plot")
    )
    state = registered(tmp_path)
    batch = state["relax_mc_batches"]["b1"]
    if change in {"partial", "failed"}:
        batch["status"] = change
    elif change == "stale":
        batch["model_version"] = "m0"
    elif change == "unverified":
        batch.pop("import_checksum")
    elif change == "plan_mismatch":
        batch["result_summary"]["plan_checksum"] = "different"
    elif change == "old_summary":
        batch["result_summary"]["model_version"] = "m0"
    else:
        state["tasks"] = [{"task_id": "t", "relax_mc_bundle_id": "b1", "status": "running"}]
    assert publish_relax_mc_analysis(state, "b1")["status"] == "not_ready"


def test_large_risk_produces_joint_evidence_not_counts_or_actions():
    report = quality_summary(sample())
    advice = report["dft_sampling_advice"]
    assert advice["decision_role"] == "supporting_evidence_alongside_ehull_and_qbc"
    assert "candidate_count" not in advice and "tool" not in advice
    assert "不是实际误差" in advice["instruction"]


def test_single_model_does_not_invent_measured_error():
    report = quality_summary(sample((0, 0.01, 0.02, 0)))
    assert "measured_energy_mae_ev_per_atom" not in report["dft_sampling_advice"]


def test_failed_render_is_explicit_and_does_not_claim_completed(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as plotting

    monkeypatch.setattr(
        plotting, "phase_diagram_reply", lambda *a, **kw: "当前 MLIP 相图未绘制：缺少数据"
    )
    assert (
        publish_relax_mc_analysis(registered(tmp_path), "b1")["status"] == "diagnostic_unavailable"
    )


def test_round_markdown_is_numbered_and_identifies_model_and_files(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as plotting
    import json

    image = tmp_path / "plot.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(plotting, "phase_diagram_reply", lambda *a, **kw: f"PNG：`{image}`")
    receipt = publish_relax_mc_analysis(registered(tmp_path), "b1")
    markdown = Path(receipt["markdown_path"]).read_text(encoding="utf-8")
    assert "b1" in markdown and "m1" in markdown and "h1" in markdown
    assert "|编号|" in markdown and str(image) in markdown
    report = json.loads(Path(receipt["review_report_path"]).read_text(encoding="utf-8"))
    assert report["bundle_id"] == "b1" and report["import_checksum"] == "receipt"


def test_missing_artifact_is_restored_but_changed_historical_hull_is_not_redrawn(
    tmp_path, monkeypatch
):
    import phase_agent.runtime.phase_diagram_chat as plotting

    image = tmp_path / "plot.png"

    def plot(*args, **kwargs):
        image.write_bytes(b"fixture")
        return f"PNG：`{image}`"

    monkeypatch.setattr(plotting, "phase_diagram_reply", plot)
    state = registered(tmp_path)
    first = publish_relax_mc_analysis(state, "b1")
    Path(first["markdown_path"]).unlink()
    assert publish_relax_mc_analysis(state, "b1")["status"] == "completed"
    Path(first["markdown_path"]).unlink()
    state["phase_diagrams"]["mlip"]["version"] = "h2"
    assert publish_relax_mc_analysis(state, "b1")["status"] == "diagnostic_unavailable"


def test_unexpected_plot_error_never_breaks_verified_import(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as plotting

    def broken(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(plotting, "phase_diagram_reply", broken)
    result = publish_relax_mc_analysis(registered(tmp_path), "b1")
    assert (
        result["status"] == "diagnostic_unavailable" and "renderer unavailable" in result["reason"]
    )
