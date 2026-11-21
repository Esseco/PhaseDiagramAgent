import json
from pathlib import Path

import pytest
from PIL import Image

from phase_agent.analysis.feedback.analysis_steps import publish_step, refresh_derived_files


def setup(tmp_path):
    root = tmp_path / "analysis_outputs"
    source = tmp_path / "source"
    source.mkdir()
    for name in ("main.png", "hd_cross.png", "selection.png"):
        Image.new("RGB", (8, 8), "white").save(source / name)
    (source / "info.json").write_text("{}", encoding="utf8")
    return root, source


def test_ordered_steps_are_idempotent_and_do_not_reorder(tmp_path):
    root, source = setup(tmp_path)
    phase = {"main.png": source / "main.png", "hd_cross.png": source / "hd_cross.png"}
    first = publish_step(
        root,
        "epoch3",
        kind="update_phasediagram",
        event_id="model:h0",
        files=phase,
        information={"hull": "h0"},
    )
    second = publish_step(
        root, "epoch3", kind="relaxmc", event_id="batch:1", files=phase, information={"hull": "h1"}
    )
    third = publish_step(
        root,
        "epoch3",
        kind="dft-selection",
        event_id="plan:1",
        files={"selection.png": source / "selection.png"},
        information={"hull": "h1"},
    )
    assert [row["directory"] for row in (first, second, third)] == [
        "0_update_phasediagram",
        "1_relaxmc-1",
        "2_dft-selection-1",
    ]
    assert all(
        (root / "epoch3/phase_diagram" / row["directory"] / "README.md").is_file()
        for row in (first, second, third)
    )
    repeated = publish_step(
        root, "epoch3", kind="relaxmc", event_id="batch:1", files=phase, information={"hull": "h1"}
    )
    assert repeated["reused"] and repeated["ordinal"] == 1
    with pytest.raises(ValueError, match="identity_changed"):
        publish_step(
            root,
            "epoch3",
            kind="relaxmc",
            event_id="batch:1",
            files=phase,
            information={"hull": "other"},
        )
    with pytest.raises(ValueError, match="order_changed"):
        publish_step(
            root,
            "epoch3",
            kind="direct",
            event_id="direct:1",
            files={"info.json": source / "info.json"},
            information={},
            expected_ordinal=1,
        )
    assert not (root / "epoch3/phase_diagram/3_direct-1").exists()


def test_steps_reject_missing_figures_or_tampered_saved_output(tmp_path):
    root, source = setup(tmp_path)
    with pytest.raises(ValueError, match="initial_hull_required"):
        publish_step(
            root,
            "epoch0",
            kind="relaxmc",
            event_id="d",
            files={"main.png": source / "main.png", "hd_cross.png": source / "hd_cross.png"},
            information={},
        )
    with pytest.raises(ValueError, match="phase_figures_required"):
        publish_step(
            root,
            "epoch0",
            kind="update_phasediagram",
            event_id="h",
            files={"main.png": source / "main.png"},
            information={},
        )
    files = {"main.png": source / "main.png", "hd_cross.png": source / "hd_cross.png"}
    row = publish_step(
        root, "epoch0", kind="update_phasediagram", event_id="h", files=files, information={}
    )
    (root / "epoch0/phase_diagram" / row["directory"] / "main.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="file_changed"):
        publish_step(
            root, "epoch0", kind="update_phasediagram", event_id="h", files=files, information={}
        )


def test_refresh_derived_files_backfills_an_old_step_without_changing_identity(tmp_path):
    root, source = setup(tmp_path)
    old = source / "voltage_curve.png"
    Image.new("RGB", (8, 8), "red").save(old)
    old_csv = source / "voltage_curve.csv"
    old_csv.write_text("status,not_ready\n", encoding="utf8")
    old_status = source / "voltage_curve_status.json"
    old_status.write_text('{"status":"not_ready"}\n', encoding="utf8")
    row = publish_step(
        root,
        "epoch0",
        kind="update_phasediagram",
        event_id="phase:m:h",
        files={
            "main.png": source / "main.png",
            "hd_cross.png": source / "hd_cross.png",
            "voltage_curve.png": old,
            "voltage_curve.csv": old_csv,
            "voltage_curve_status.json": old_status,
        },
        information={"hull": "h"},
    )
    Image.new("RGB", (8, 8), "blue").save(old)
    old_csv.write_text("status,completed\n", encoding="utf8")
    old_status.write_text('{"status":"completed"}\n', encoding="utf8")
    refreshed = refresh_derived_files(
        root,
        "epoch0",
        row,
        {
            "voltage_curve.png": old,
            "voltage_curve.csv": old_csv,
            "voltage_curve_status.json": old_status,
        },
    )
    assert refreshed["event_id"] == row["event_id"]
    assert refreshed["directory"] == row["directory"]
    assert refreshed["derived_refreshed"] is True
    target = root / "epoch0" / "phase_diagram" / row["directory"]
    with Image.open(target / "voltage_curve.png") as image:
        assert image.getpixel((0, 0))[:3] == (0, 0, 255)
    assert (
        json.loads((target / "manifest.json").read_text(encoding="utf8"))["files"]
        == refreshed["files"]
    )


def test_registered_events_follow_publication_order_and_replay(tmp_path):
    from phase_agent.analysis.feedback.analysis_events import (
        phase_event,
        relax_event,
        selection_event,
    )
    from phase_agent.analysis.feedback.versioned_products import publish_bundle

    root, source = setup(tmp_path)
    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 1}}}

    def build(folder):
        for name in ("main.png", "hd_cross.png"):
            Image.new("RGB", (8, 8), "white").save(folder / name)
        (folder / "phase_diagram.json").write_text('{"version":"h"}', encoding="utf8")
        (folder / "report.md").write_text("hull", encoding="utf8")

    publication = publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch0",
        model="m",
        identity={"hull_version": "h"},
        required=("main.png", "hd_cross.png", "phase_diagram.json", "report.md"),
        build=build,
    )
    first = phase_event(state, root, publication)
    receipt = {
        "status": "completed",
        "model_version": "m",
        "import_checksum": "abc",
        "hull_version": "h2",
        "plot_paths": [str(source / "main.png"), str(source / "hd_cross.png")],
    }
    state["relax_mc_batches"] = {
        "batch": {"status": "completed", "model_version": "m", "import_checksum": "abc"}
    }
    second = relax_event(state, root, "batch", receipt)

    def build_next(folder):
        build(folder)
        (folder / "phase_diagram.json").write_text('{"version":"h2"}', encoding="utf8")

    next_publication = publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch0",
        model="m",
        identity={"hull_version": "h2"},
        required=("main.png", "hd_cross.png", "phase_diagram.json", "report.md"),
        build=build_next,
    )
    assert phase_event(state, root, next_publication)["ordinal"] == 1
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf8")
    rows = [
        {
            "status": "completed",
            "model_version": "m",
            "hull_version": "h",
            "selected_count": 1,
            "path": str(source / "selection.png"),
        }
    ]
    third = selection_event(state, root, plan, rows)
    assert [x["directory"] for x in (first, second, third)] == [
        "0_update_phasediagram",
        "1_relaxmc-1",
        "2_dft-selection-1",
    ]
    assert relax_event(state, root, "batch", receipt)["ordinal"] == 1
    assert phase_event(state, root, publication)["ordinal"] == 0
    assert selection_event(state, root, plan, rows)["ordinal"] == 2
    assert "earlier records remain" in (
        root / "epoch0/phase_diagram/0_update_phasediagram/README.md"
    ).read_text(encoding="utf8")
    state["active_model_version"] = "other"
    with pytest.raises(ValueError, match="binding_invalid"):
        relax_event(state, root, "other", receipt)


def test_direct_can_start_without_fabricating_phase(tmp_path):
    from hashlib import sha256
    from phase_agent.analysis.feedback.analysis_events import direct_event

    root, source = setup(tmp_path)
    plan = source / "selection.md"
    plan.write_text("selected structures", encoding="utf8")
    receipt = {
        "model_version": "m",
        "scope_id": "batch",
        "source_digest": "123",
        "selected_count": 1,
        "plan_path": str(plan),
        "plan_sha256": sha256(plan.read_bytes()).hexdigest(),
    }
    manifest = source / "direct_result.json"
    manifest.write_text(json.dumps(receipt), encoding="utf8")
    receipt["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 1}}}
    step = direct_event(state, root, receipt)
    assert step["directory"] == "0_direct-1"
    assert "No verified current phase" in step["information"]["phase_context"]
    assert direct_event(state, root, receipt)["ordinal"] == 0


def test_direct_does_not_publish_stale_phase_when_status_is_not_ready(tmp_path):
    from hashlib import sha256
    from phase_agent.analysis.feedback.analysis_events import direct_event
    from phase_agent.analysis.feedback.versioned_products import publish_bundle, publication_status

    root, source = setup(tmp_path)
    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 1}}}

    def build(folder):
        for name in ("main.png", "hd_cross.png"):
            Image.new("RGB", (8, 8), "white").save(folder / name)
        (folder / "phase_diagram.json").write_text('{"version":"h"}', encoding="utf8")
        (folder / "report.md").write_text("hull", encoding="utf8")

    publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch0",
        model="m",
        identity={"hull_version": "h"},
        required=("main.png", "hd_cross.png", "phase_diagram.json", "report.md"),
        build=build,
    )
    publication_status(
        root, "phase_diagram", "m", "epoch0", status="not_ready", reason="quarantined"
    )
    plan = source / "selection.md"
    plan.write_text("direct", encoding="utf8")
    receipt = {
        "model_version": "m",
        "scope_id": "batch",
        "source_digest": "456",
        "selected_count": 1,
        "plan_path": str(plan),
        "plan_sha256": sha256(plan.read_bytes()).hexdigest(),
    }
    manifest = source / "direct_result.json"
    manifest.write_text(json.dumps(receipt), encoding="utf8")
    receipt["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    assert direct_event(state, root, receipt)["directory"] == "0_direct-1"


def test_accepted_threshold_stop_can_publish_but_incomplete_return_cannot(tmp_path, monkeypatch):
    from phase_agent.analysis.feedback import analysis_events

    root, source = setup(tmp_path)
    receipt = {
        "status": "completed",
        "model_version": "m",
        "import_checksum": "abc",
        "hull_version": "h",
        "plot_paths": [str(source / "main.png"), str(source / "hd_cross.png")],
    }
    summary = {
        "status": "paused",
        "reason": "continuation_pool_below_minimum",
        "completed_mc_waves": [0],
        "task_count": 1,
        "failed_task_ids": [],
    }
    task = {"task_id": "t", "status": "completed", "relax_mc_bundle_id": "batch"}
    state = {
        "active_model_version": "m",
        "upload_layout": {"model_rounds": {"m": 1}},
        "tasks": [task],
        "relax_mc_batches": {
            "batch": {
                "status": "paused",
                "formal_recovery_status": "accepted",
                "model_version": "m",
                "import_checksum": "abc",
                "result_summary": summary,
            }
        },
    }
    monkeypatch.setattr(analysis_events, "phase_event", lambda *args: None)
    monkeypatch.setattr(
        analysis_events, "publish_step", lambda *args, **kwargs: {"event_id": kwargs["event_id"]}
    )
    assert analysis_events.relax_event(state, root, "batch", receipt)["event_id"] == "relaxmc:batch"
    task["status"] = "pending"
    with pytest.raises(ValueError, match="analysis_step_relax_binding_invalid"):
        analysis_events.relax_event(state, root, "batch", receipt)
