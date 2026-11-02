"""Synthetic publication regression; no real project, model or HPC access."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.analysis.feedback.analysis_publication import publish_analysis_products
from phase_agent.analysis.feedback.versioned_products import publish_bundle, validate_bundle
from tests.test_official_dft_products import result, predict
from tests.test_phase_quality import sample


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def phase_state():
    state = sample()
    state["upload_layout"] = {"model_rounds": {"m1": 1, "m2": 2}}
    for row in state["phase_diagrams"]["mlip"]["entries"]:
        row.update(ehull=0, is_stable=True)
    return state


def files(directory):
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in Path(directory).rglob("*")
        if p.is_file()
    }


def test_real_phase_pair_version_reuse_increment_and_failed_hd(tmp_path, monkeypatch):
    state = phase_state()
    publish_analysis_products(state, tmp_path)
    publication = state["analysis_publications"]["phase_diagram"]
    assert publication["status"] == "completed", publication
    archive = Path(publication["directory"])
    original = files(archive)
    from PIL import Image

    for name in ("main.png", "hd_cross.png"):
        with Image.open(archive / name) as image:
            assert image.format == "PNG" and min(image.size) > 500
    first_time = (archive / "manifest.json").stat().st_mtime_ns
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["phase_diagram"] == publication
    assert (archive / "manifest.json").stat().st_mtime_ns == first_time
    assert files(archive) == original
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] -= 0.01
    state["phase_diagrams"]["mlip"]["version"] = "h2"
    publish_analysis_products(state, tmp_path)
    newer = state["analysis_publications"]["phase_diagram"]
    assert newer["status"] == "completed" and newer["version"] != publication["version"]
    assert validate_bundle(tmp_path / "latest" / "phase_diagram")["version"] == newer["version"]
    assert files(archive) == original
    before = files(tmp_path / "latest" / "phase_diagram")
    import phase_agent.analysis.phase.phase_quality_comparison as hd

    monkeypatch.setattr(
        hd,
        "publish_quality_comparison",
        lambda *a, **kw: {"status": "unavailable", "reason": "injected HD failure"},
    )
    state["phase_diagrams"]["mlip"]["version"] = "h3"
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["phase_diagram"]["status"] == "not_ready"
    assert "injected HD failure" in state["analysis_publications"]["phase_diagram"]["reason"]
    assert files(tmp_path / "latest" / "phase_diagram") == before
    assert load(tmp_path / "latest" / "phase_diagram_status.json")["status"] == "not_ready"
    assert len(list((tmp_path / "epoch0" / "phase_diagram" / "versions").iterdir())) == 2
    assert not list(tmp_path.rglob(".build-*")) and not list(tmp_path.rglob(".latest-*"))


def add_dft(state, task, operation, energy=-7.0, force=0.2, model="m1"):
    from phase_agent.analysis.feedback.dft_result_products import record_dft_products

    row = deepcopy(result())
    row.update(task_id=task, model_version=model)
    state.setdefault("tasks", []).append(
        {"task_id": task, "upload_operation_id": operation, "model_version": model}
    )
    record_dft_products(
        state,
        row,
        lambda **kw: {
            **predict(),
            "energy": energy,
            "forces": [[force] * 3] * 2,
            "model_version": model,
        },
    )


def test_dft_epoch_cumulative_dedup_original_model_and_latest(tmp_path):
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1, "m2": 2}}}
    add_dft(state, "T1", "round1")
    publish_analysis_products(state, tmp_path)
    first = deepcopy(state["analysis_publications"]["dft_error"])
    assert first["status"] == "completed", first
    original = files(Path(first["directory"]))
    add_dft(state, "T2", "round2", energy=-5.0, force=0.4)
    # Duplicate receipt must not count a frame twice.
    state["dft_dataset_records"].append(deepcopy(state["dft_dataset_records"][0]))
    publish_analysis_products(state, tmp_path)
    second = state["analysis_publications"]["dft_error"]
    assert second["status"] == "completed", second
    metrics = load(Path(second["directory"]) / "metrics.json")
    assert metrics["matched_structures"] == 2
    assert metrics["energy_total"]["mae"] == 2
    assert metrics["energy_total"]["rmse"] == pytest.approx(5**0.5)
    assert metrics["forces"]["rmse"] == pytest.approx(0.1**0.5)
    assert files(Path(first["directory"])) == original
    stamp = (Path(second["directory"]) / "manifest.json").stat().st_mtime_ns
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["dft_error"] == second
    assert (Path(second["directory"]) / "manifest.json").stat().st_mtime_ns == stamp
    state["active_model_version"] = "m2"
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["dft_error"]["status"] == "not_ready"
    assert load(tmp_path / "latest" / "dft_error_status.json")["model_version"] == "m2"
    assert validate_bundle(tmp_path / "latest" / "dft_error")["version"] == second["version"]
    add_dft(state, "T3", "round3", model="m2", energy=-6.0)
    publish_analysis_products(state, tmp_path)
    third = state["analysis_publications"]["dft_error"]
    assert third["status"] == "completed" and third["epoch"] == "epoch1"
    assert load(Path(third["directory"]) / "metrics.json")["matched_structures"] == 1
    assert (
        validate_bundle(tmp_path / "epoch0" / "dft_error" / "latest")["version"]
        == second["version"]
    )
    assert validate_bundle(tmp_path / "latest" / "dft_error")["version"] == third["version"]


def test_model_switch_marks_old_phase_unavailable_without_relabeling(tmp_path):
    from phase_agent.persistence.models.mark_model_dependent_results_stale import (
        mark_model_dependent_results_stale,
    )

    state = phase_state()
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["phase_diagram"]["status"] == "completed"
    before = files(tmp_path / "latest" / "phase_diagram")
    updated = mark_model_dependent_results_stale(
        state, old_model_version="m1", new_model_version="m2"
    )["state"]
    assert updated["analysis_publications"]["phase_diagram"]["status"] == "not_ready"
    assert load(tmp_path / "latest" / "phase_diagram_status.json")["model_version"] == "m2"
    assert files(tmp_path / "latest" / "phase_diagram") == before
    assert not list((tmp_path / "epoch1").rglob("main.png"))
    fresh = updated["phase_diagrams"]["mlip"]
    fresh.update(model_version="m2", version="new-model-hull", validity="valid")
    for row in fresh["entries"]:
        row["source_version"] = "m2"
    publish_analysis_products(updated, tmp_path)
    current = updated["analysis_publications"]["phase_diagram"]
    assert current["status"] == "completed" and current["epoch"] == "epoch1"
    assert (
        validate_bundle(tmp_path / "latest" / "phase_diagram")["identity"]["model_version"] == "m2"
    )
    assert files(tmp_path / "epoch0" / "phase_diagram" / "latest") == before


def test_archive_integrity_and_complete_bundle_failure_keep_latest(tmp_path, monkeypatch):
    def build(folder):
        from PIL import Image

        for name in ("main.png", "hd_cross.png"):
            Image.new("RGB", (2, 2)).save(folder / name)

    kwargs = dict(
        kind="phase_diagram",
        epoch="epoch0",
        model="m1",
        required=("main.png", "hd_cross.png"),
        build=build,
    )
    first = publish_bundle(tmp_path, identity={"input": 1}, **kwargs)
    before = files(tmp_path / "latest" / "phase_diagram")
    import phase_agent.analysis.feedback.versioned_products as module

    original_copy = module.shutil.copytree

    def fail_global(src, dst, *args, **kw):
        if Path(dst).parent.parent == tmp_path / "latest":
            raise OSError("injected copy failure")
        return original_copy(src, dst, *args, **kw)

    monkeypatch.setattr(module.shutil, "copytree", fail_global)
    with pytest.raises(OSError, match="injected"):
        publish_bundle(tmp_path, identity={"input": 2}, **kwargs)
    assert files(tmp_path / "latest" / "phase_diagram") == before
    monkeypatch.setattr(module.shutil, "copytree", original_copy)
    (Path(first["directory"]) / "main.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        publish_bundle(tmp_path, identity={"input": 1}, **kwargs)


def test_missing_original_prediction_stays_pending(tmp_path):
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1}}}
    add_dft(state, "T1", "round1")
    state["dft_dataset_records"][0]["mlip_prediction"]["model_version"] = "wrong-model"
    publish_analysis_products(state, tmp_path)
    assert state["analysis_publications"]["dft_error"]["status"] == "not_ready"
    assert not list(tmp_path.rglob("energy_parity.png"))


def test_default_only_mlip_and_legacy_dft_is_explicit(tmp_path):
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from phase_agent.analysis.feedback.output_catalog import output_catalog
    from tests.test_na_eform_phase_csv import record

    records = [record("left", 0, -3), record("right", 1, -4)]
    for row in records:
        row["energy_method"] = "dft"
    ordinary = update_phase_diagram(records, output_directory=tmp_path)
    assert "dft" not in ordinary["diagrams"]
    assert not list(tmp_path.rglob("*dft*"))
    legacy = update_phase_diagram(records, output_directory=tmp_path, include_dft=True)
    assert legacy["diagrams"]["dft"]["status"] == "completed"
    state = {"phase_diagrams": legacy["diagrams"]}
    assert not any(row["product"] == "phase_diagram_dft" for row in output_catalog(state))
    assert Path(legacy["diagrams"]["dft"]["csv_path"]).is_file()


def test_existing_export_entry_updates_latest_and_catalog(tmp_path):
    from phase_agent.analysis.feedback.export_dft_products import export_dft_products

    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1}}}
    add_dft(state, "T1", "round1")
    export_dft_products(state, tmp_path)
    assert state["analysis_publications"]["dft_error"]["status"] == "completed"
    assert (tmp_path / "latest" / "dft_error" / "energy_parity.png").is_file()
    index = (tmp_path / "output_index.md").read_text(encoding="utf-8")
    assert "dft_error latest" in index and "dft_error versions" in index
    assert state["dft_result_exports"]


def test_conflicting_dataset_or_model_fingerprint_keeps_last_complete(tmp_path):
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1}}}
    add_dft(state, "T1", "round1")
    publish_analysis_products(state, tmp_path)
    before = files(tmp_path / "latest" / "dft_error")
    duplicate = deepcopy(state["dft_dataset_records"][0])
    duplicate["energy"] += 1
    state["dft_dataset_records"].append(duplicate)
    publish_analysis_products(state, tmp_path)
    assert (
        "Conflicting DFT dataset identity" in state["analysis_publications"]["dft_error"]["reason"]
    )
    assert files(tmp_path / "latest" / "dft_error") == before
    state["dft_dataset_records"].pop()
    state["model_registry"] = {"m1": {"comparison_binding_error": "model bytes changed"}}
    publish_analysis_products(state, tmp_path)
    assert "model bytes changed" in state["analysis_publications"]["dft_error"]["reason"]
    assert files(tmp_path / "latest" / "dft_error") == before


def test_round_never_accepts_main_without_hd(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as plotting
    from phase_agent.analysis.phase.phase_round_analysis import publish_relax_mc_analysis
    from tests.test_phase_round_analysis import registered

    image = tmp_path / "only-main.png"
    image.write_bytes(b"synthetic-stand-in")

    def plot(*args, **kwargs):
        kwargs["artifacts"]["main"] = [str(image)]
        return "main only"

    monkeypatch.setattr(plotting, "_plot", plot)
    assert (
        publish_relax_mc_analysis(registered(tmp_path), "b1")["status"] == "diagnostic_unavailable"
    )


def test_latest_swap_rollback_restores_previous_copy(tmp_path, monkeypatch):
    from PIL import Image

    def build(folder):
        for name in ("main.png", "hd_cross.png"):
            Image.new("RGB", (2, 2)).save(folder / name)

    kwargs = dict(
        kind="phase_diagram",
        epoch="epoch0",
        model="m1",
        required=("main.png", "hd_cross.png"),
        build=build,
    )
    publish_bundle(tmp_path, identity={"input": 1}, **kwargs)
    before = files(tmp_path / "epoch0" / "phase_diagram" / "latest")
    original = Path.rename

    def fail_rename(source, target):
        if source.name == "new" and source.parent.name.startswith(".latest-"):
            raise OSError("injected rename failure")
        return original(source, target)

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="injected rename"):
        publish_bundle(tmp_path, identity={"input": 2}, **kwargs)
    assert files(tmp_path / "epoch0" / "phase_diagram" / "latest") == before
    assert files(tmp_path / "latest" / "phase_diagram") == before
    assert load(tmp_path / "latest" / "phase_diagram_status.json")["status"] == "not_ready"
    assert not list(tmp_path.rglob(".latest-*"))
