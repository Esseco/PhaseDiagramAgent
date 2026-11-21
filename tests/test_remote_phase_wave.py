import json
from pathlib import Path
import pytest
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.phase_analysis_wave import analyze_wave


def fixture_wave(root):
    rows = []
    for name in ("one", "two"):
        folder = root / name
        folder.mkdir()
        structure = folder / "final.vasp"
        structure.write_text(name)
        result = {
            "task_id": name,
            "task_key": name,
            "stage": "relax_and_feature",
            "status": "completed",
            "task_checksum": "input-" + name,
            "outputs": {
                "structure_path": str(structure),
                "structure_checksum": file_checksum(structure),
            },
        }
        result_path = folder / "result.json"
        result_path.write_text(json.dumps(result))
        (folder / "task.finished.json").write_text(
            json.dumps(
                {
                    "task_id": name,
                    "task_key": name,
                    "status": "completed",
                    "task_checksum": "input-" + name,
                    "result_checksum": file_checksum(result_path),
                }
            )
        )
        rows.append(
            {
                "task_id": name,
                "task_key": name,
                "task_checksum": "input-" + name,
                "result_path": f"{name}/result.json",
                "branch_id": "B",
            }
        )
    return {
        "manager": {"boundary": {}, "branches": {}, "structures": {}, "system_config": {}},
        "tasks": rows,
    }


def classify(result, manager, **kwargs):
    digest = file_checksum(result["outputs"]["structure_path"])
    return {
        "outputs": {
            "phase_identification": {
                "status": "unknown",
                "reason": "ValueError: unrecognized_phase:X",
                "structure_sha256": digest,
                "portable_classification_scope": "rules",
            }
        }
    }


def test_wave_waits_for_every_receipt_and_preserves_signed_results(tmp_path):
    spec = fixture_wave(tmp_path)
    signed = (tmp_path / "one/result.json").read_bytes()
    (tmp_path / "two/task.finished.json").unlink()
    with pytest.raises(ValueError, match="unverified_result"):
        analyze_wave(tmp_path, spec, classify=classify)
    assert not (tmp_path / "one/phase_identification.json").exists()
    result = json.loads((tmp_path / "two/result.json").read_text())
    (tmp_path / "two/task.finished.json").write_text(
        json.dumps(
            {
                "task_id": "two",
                "task_key": "two",
                "status": "completed",
                "task_checksum": "input-two",
                "result_checksum": file_checksum(tmp_path / "two/result.json"),
            }
        )
    )
    assert result["task_id"] == "two"
    assert analyze_wave(tmp_path, spec, classify=classify)["task_ids"] == ["one", "two"]
    assert (tmp_path / "one/result.json").read_bytes() == signed
    sidecar = json.loads((tmp_path / "one/phase_identification.json").read_text())
    assert sidecar["result_checksum"] == file_checksum(tmp_path / "one/result.json")
    assert sidecar["phase_identification"]["status"] == "unknown"
    assert sidecar["phase_identification"]["origin"] == "remote_cpu"
    assert analyze_wave(tmp_path, spec, classify=classify)["task_ids"] == ["one", "two"]
    sidecar_path = tmp_path / "one/phase_identification.json"
    sidecar["phase_identification"]["reason"] = "conflicting"
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="existing_evidence_conflict"):
        analyze_wave(tmp_path, spec, classify=classify)
    assert json.loads(sidecar_path.read_text())["phase_identification"]["reason"] == "conflicting"


def test_sidecar_is_bound_to_exact_task_result_and_structure(tmp_path):
    from phase_agent.tools.remote.phase_result_evidence import verified_phase_evidence

    spec = fixture_wave(tmp_path)
    analyze_wave(tmp_path, spec, classify=classify)
    result_path = tmp_path / "one/result.json"
    task = spec["tasks"][0]
    result = json.loads(result_path.read_text())
    assert verified_phase_evidence(
        result_path.with_name("phase_identification.json"), result_path, task, result
    )
    (tmp_path / "one/final.vasp").write_text("changed")
    assert (
        verified_phase_evidence(
            result_path.with_name("phase_identification.json"), result_path, task, result
        )
        is None
    )


def test_refresh_cpu_handoff_depends_on_all_gpu_batches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.tools.remote.prepare_phase_analysis import prepare_phase_analysis
    import phase_agent.tools.remote.submission_handoff as handoff

    root = tmp_path / "epoch1"
    batches = []
    for index in range(2):
        folder = root / "calculations" / f"operation{index}" / "inputs" / "batch"
        folder.mkdir(parents=True)
        job = folder / "GPU.sh"
        job.write_text("#!/bin/bash\ntrue\n", encoding="utf8", newline="\n")
        task = {
            "task_id": f"t{index}",
            "task_key": f"key{index}",
            "stage": "relax_and_feature",
            "status": "pending",
            "branch_id": "B",
            "input_structure_sha256": "x",
            "task_checksum": f"c{index}",
        }
        (folder / "task.json").write_text(json.dumps(task))
        entry = {**task, "input_path": "task.json", "result_path": "../../results/t/result.json"}
        (folder / "manifest.json").write_text(json.dumps([entry]))
        batches.append(
            {
                "batch_id": f"batch{index}",
                "upload_directory": str(folder),
                "script_path": str(job),
                "results_directory": str(folder.parent.parent / "results"),
                "task_ids": [f"t{index}"],
                "status": "prepared",
            }
        )
    state = {
        "tasks": [{"task_id": f"t{i}", "status": "pending"} for i in range(2)],
        "slurm_batches": batches,
    }
    manager = SimpleNamespace(boundary={}, data={"system_config": {}})
    profiles = {"gpu": {"stages": ["relax_and_feature"], "slurm_options": {}, "shell_preamble": []}}
    metadata = prepare_phase_analysis(state, batches, manager, {}, profiles)
    assert set(metadata["batch_ids"]) == {"batch0", "batch1"}
    cpu = Path(metadata["script_path"]).read_text()
    assert "--partition=p1" in cpu and "--gres" not in cpu
    assert 'cd -- "${SLURM_SUBMIT_DIR:?' in cpu
    assert 'dirname "$0"' not in cpu
    import os
    import subprocess
    import sys

    runtime = Path(metadata["script_path"]).parent / "runtime"
    env = {**os.environ, "PYTHONPATH": str(runtime), "PYTHONNOUSERSITE": "1"}
    check = subprocess.run(
        [sys.executable, "-m", "phase_agent.tools.remote.phase_analysis_wave", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert check.returncode == 0, check.stderr

    from pymatgen.core import Lattice, Structure

    isolated_root = tmp_path / "isolated_results"
    isolated_root.mkdir()
    isolated_spec = fixture_wave(isolated_root)
    isolated_spec["manager"]["boundary"] = {"P": ["O3"]}
    structure = Structure(
        Lattice.cubic(4), ["Na", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    )
    for row in isolated_spec["tasks"]:
        folder = isolated_root / row["task_id"]
        structure.to(filename=folder / "final.vasp", fmt="poscar")
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        result["outputs"]["structure_checksum"] = file_checksum(folder / "final.vasp")
        result_path.write_text(json.dumps(result))
        marker_path = folder / "task.finished.json"
        marker = json.loads(marker_path.read_text())
        marker["result_checksum"] = file_checksum(result_path)
        marker_path.write_text(json.dumps(marker))
    spec_path = isolated_root / "spec.json"
    spec_path.write_text(json.dumps(isolated_spec), encoding="utf8")
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "phase_agent.tools.remote.phase_analysis_wave",
            str(spec_path),
            str(isolated_root),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert check.returncode == 0, check.stderr
    assert (
        json.loads((isolated_root / "one/phase_identification.json").read_text())[
            "phase_identification"
        ]["phase"]
        == "O3"
    )
    captured = {}
    actual_prepare = handoff.prepare_handoff

    def fake_prepare(directory, scripts, **kwargs):
        captured.update(directory=directory, scripts=scripts, **kwargs)
        return {"upload_archive": "archive"}

    monkeypatch.setattr(handoff, "prepare_handoff", fake_prepare)
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_storage.restore_storage_directory", lambda *args: None
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.task_waiting.awaiting_task_result", lambda task: True
    )
    # A previously sent sibling must never be silently omitted from the wave.
    batches[1]["status"] = "submitted"
    with pytest.raises(ValueError, match="phase_analysis_incomplete_handoff_scope"):
        handoff.prepare_batch_handoff(state, batches[0])
    batches[1]["status"] = "prepared"
    handoff.prepare_batch_handoff(state, batches[0])
    cpu_key, parents = next(iter(captured["afterany"].items()))
    assert len(parents) == 2 and cpu_key.endswith("CPU.sh")
    assert len(captured["extra_result_roots"][cpu_key]) == 2

    # Exercise the real archive builder and production admission checks.
    import zipfile

    package = actual_prepare(
        captured["directory"],
        captured["scripts"],
        afterany=captured["afterany"],
        extra_result_roots=captured["extra_result_roots"],
        publish_on_success=captured.get("publish_on_success"),
    )
    with zipfile.ZipFile(package["upload_archive"]) as archive:
        plan = json.loads(archive.read("submission_plan.json"))
        assert plan["afterany"][cpu_key] == parents
        cpu_script = archive.read(cpu_key).decode("utf8")
        assert cpu_script.count("_round_transfer.py") == 2
        assert any(name.endswith("phase_wave.json") for name in archive.namelist())
        assert any(name.endswith("phase_analysis_wave.py") for name in archive.namelist())


def test_cpu_wave_uses_real_rule_and_result_bundle(tmp_path):
    from pymatgen.core import Lattice, Structure
    from phase_agent.tools.remote.transfer_bundle import result_bundle, validate_bundle
    from phase_agent.tools.remote.phase_result_evidence import verified_phase_evidence
    import zipfile

    spec = fixture_wave(tmp_path)
    spec["manager"]["boundary"] = {"P": ["O3"]}
    structure = Structure(
        Lattice.cubic(4), ["Na", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    )
    for row in spec["tasks"]:
        folder = tmp_path / row["task_id"]
        structure.to(filename=folder / "final.vasp", fmt="poscar")
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        result["outputs"]["structure_checksum"] = file_checksum(folder / "final.vasp")
        result_path.write_text(json.dumps(result))
        marker_path = folder / "task.finished.json"
        marker = json.loads(marker_path.read_text())
        marker["result_checksum"] = file_checksum(result_path)
        marker_path.write_text(json.dumps(marker))
    assert len(analyze_wave(tmp_path, spec)["task_ids"]) == 2
    sidecar = tmp_path / "one/phase_identification.json"
    payload = json.loads(sidecar.read_text())
    assert payload["phase_identification"]["phase"] == "O3"
    result_path = tmp_path / "one/result.json"
    assert verified_phase_evidence(
        sidecar, result_path, spec["tasks"][0], json.loads(result_path.read_text())
    )
    archive = result_bundle(tmp_path, [tmp_path / "one"], "one")
    assert validate_bundle(archive) > 0
    with zipfile.ZipFile(archive) as zipped:
        assert "one/phase_identification.json" in zipped.namelist()


def test_cpu_single_point_rejects_changed_input_structure(tmp_path):
    from pymatgen.core import Lattice, Structure

    spec = fixture_wave(tmp_path)
    row = spec["tasks"][0]
    folder = tmp_path / row["task_id"]
    Structure(
        Lattice.cubic(4), ["Na", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    ).to(filename=folder / "final.vasp", fmt="poscar")
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text())
    result["outputs"].update(
        structure_checksum=file_checksum(folder / "final.vasp"), single_point_completed=True
    )
    result_path.write_text(json.dumps(result))
    marker_path = folder / "task.finished.json"
    receipt = json.loads(marker_path.read_text())
    receipt["result_checksum"] = file_checksum(result_path)
    marker_path.write_text(json.dumps(receipt))
    row["input_structure_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="phase_analysis_single_point_input_mismatch"):
        analyze_wave(tmp_path, spec)


def test_cpu_single_point_inherits_and_missing_source_is_identified(tmp_path):
    from pymatgen.core import Lattice, Structure
    from types import SimpleNamespace
    from phase_agent.analysis.phase.identify_result_phase import identify_result_phase

    spec = fixture_wave(tmp_path)
    spec["manager"]["boundary"] = {"P": ["O3"]}
    structure = Structure(
        Lattice.cubic(4), ["Na", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    )
    for row in spec["tasks"]:
        folder = tmp_path / row["task_id"]
        structure.to(filename=folder / "final.vasp", fmt="poscar")
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        digest = file_checksum(folder / "final.vasp")
        result["outputs"].update(structure_checksum=digest, single_point_completed=True)
        result_path.write_text(json.dumps(result))
        marker_path = folder / "task.finished.json"
        marker = json.loads(marker_path.read_text())
        marker["result_checksum"] = file_checksum(result_path)
        marker_path.write_text(json.dumps(marker))
        row["input_structure_sha256"] = digest
    manager = SimpleNamespace(boundary=spec["manager"]["boundary"], data=spec["manager"])
    source = identify_result_phase(
        {
            "stage": "relax_and_feature",
            "status": "completed",
            "outputs": {"structure_path": str(tmp_path / "one/final.vasp")},
        },
        manager,
    )
    spec["tasks"][0]["source_phase"] = source["outputs"]["phase_identification"]

    classified = []

    def classify_missing(result, manager, **kwargs):
        classified.append(result["task_id"])
        return identify_result_phase(result, manager, **kwargs)

    analyze_wave(tmp_path, spec, classify=classify_missing)
    first = json.loads((tmp_path / "one/phase_identification.json").read_text())[
        "phase_identification"
    ]
    second = json.loads((tmp_path / "two/phase_identification.json").read_text())[
        "phase_identification"
    ]
    assert first["phase"] == "O3"
    assert classified == ["two"]
    assert second["status"] == "identified" and second["phase"] == "O3"


def test_each_relaxation_is_reidentified_even_with_prior_phase(tmp_path):
    spec = fixture_wave(tmp_path)
    seen = []
    for row in spec["tasks"]:
        path = tmp_path / row["result_path"]
        result = json.loads(path.read_text())
        result["outputs"]["phase_identification"] = {"status": "identified", "phase": "old"}
        path.write_text(json.dumps(result))
        marker_path = path.with_name("task.finished.json")
        marker = json.loads(marker_path.read_text())
        marker["result_checksum"] = file_checksum(path)
        marker_path.write_text(json.dumps(marker))

    def fresh(result, manager, **kwargs):
        assert "phase_identification" not in result["outputs"]
        assert kwargs["cache"] is None
        seen.append(result["task_id"])
        return classify(result, manager, **kwargs)

    analyze_wave(tmp_path, spec, classify=fresh)
    assert seen == ["one", "two"]


def test_cpu_groups_require_every_bound_receipt(tmp_path):
    from phase_agent.tools.remote.phase_analysis_wave import write_group_receipt, verify_groups

    rows = []
    for index in range(201):
        folder = tmp_path / str(index)
        folder.mkdir()
        structure = folder / "final.vasp"
        structure.write_text(str(index))
        result = {
            "task_id": str(index),
            "task_key": str(index),
            "task_checksum": str(index),
            "stage": "relax_and_feature",
            "status": "completed",
            "outputs": {
                "structure_path": str(structure),
                "structure_checksum": file_checksum(structure),
            },
        }
        target = folder / "result.json"
        target.write_text(json.dumps(result))
        (folder / "task.finished.json").write_text(
            json.dumps(
                {
                    "task_id": str(index),
                    "task_key": str(index),
                    "task_checksum": str(index),
                    "status": "completed",
                    "result_checksum": file_checksum(target),
                }
            )
        )
        rows.append(
            {
                "task_id": str(index),
                "task_key": str(index),
                "task_checksum": str(index),
                "result_path": f"{index}/result.json",
            }
        )
    spec = {
        "manager": {"boundary": {}, "branches": {}, "structures": {}, "system_config": {}},
        "tasks": rows,
    }
    path = tmp_path / "phase_wave.json"
    path.write_text(json.dumps(spec))
    receipt = write_group_receipt(tmp_path, path, 0, classify=classify)
    assert len(receipt["scope"]) == 200 and len(receipt["artifacts"]) == 200
    with pytest.raises(ValueError, match="receipt_missing"):
        verify_groups(tmp_path, path)
    receipt = write_group_receipt(tmp_path, path, 1, classify=classify)
    assert receipt["scope"] == ["200"]
    assert verify_groups(tmp_path, path)["group_count"] == 2
    (tmp_path / "200/phase_identification.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence_invalid"):
        verify_groups(tmp_path, path)


def test_large_refresh_creates_cpu_array_and_finalizer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.tools.remote.prepare_phase_analysis import prepare_phase_analysis
    import phase_agent.tools.remote.submission_handoff as handoff

    root = tmp_path / "epoch1"
    folder = root / "calculations/op/inputs/batch"
    folder.mkdir(parents=True)
    job = folder / "GPU.sh"
    job.write_text("#!/bin/bash\ntrue\n", encoding="utf8", newline="\n")
    task = {"branch_id": "B", "input_structure_sha256": "digest"}
    (folder / "task.json").write_text(json.dumps(task))
    entries = [
        {
            "task_id": str(i),
            "task_key": str(i),
            "input_path": "task.json",
            "result_path": f"../../results/{i}/result.json",
        }
        for i in range(201)
    ]
    (folder / "manifest.json").write_text(json.dumps(entries))
    batch = {
        "batch_id": "b",
        "upload_directory": str(folder),
        "script_path": str(job),
        "results_directory": str(folder.parent.parent / "results"),
        "task_ids": [str(i) for i in range(201)],
        "status": "prepared",
    }
    state = {
        "tasks": [{"task_id": str(i), "status": "pending"} for i in range(201)],
        "slurm_batches": [batch],
    }
    small_state = {
        "tasks": state["tasks"][:200],
        "slurm_batches": [dict(batch, task_ids=batch["task_ids"][:200])],
    }
    (folder / "manifest.json").write_text(json.dumps(entries[:200]))
    small = prepare_phase_analysis(
        small_state,
        small_state["slurm_batches"],
        SimpleNamespace(boundary={}, data={"system_config": {}}),
        {},
        {"gpu": {"stages": ["relax_and_feature"], "slurm_options": {}}},
    )
    assert "group_count" not in small and "--array" not in Path(small["script_path"]).read_text()
    (folder / "manifest.json").write_text(json.dumps(entries))
    meta = prepare_phase_analysis(
        state,
        [batch],
        SimpleNamespace(boundary={}, data={"system_config": {}}),
        {},
        {"gpu": {"stages": ["relax_and_feature"], "slurm_options": {}}},
    )
    assert meta["group_count"] == 2
    assert "#SBATCH --array=0-1" in Path(meta["script_path"]).read_text()
    captured = {}
    actual_prepare = handoff.prepare_handoff
    monkeypatch.setattr(
        "phase_agent.tools.state.task_waiting.awaiting_task_result", lambda task: True
    )
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_storage.restore_storage_directory", lambda *args: None
    )
    monkeypatch.setattr(
        handoff,
        "prepare_handoff",
        lambda directory, scripts, **kw: (
            captured.update(scripts=scripts, **kw) or {"upload_archive": "archive"}
        ),
    )
    handoff.prepare_batch_handoff(state, batch)
    cpu = next(k for k in captured["afterany"] if k.endswith("CPU.sh"))
    final = next(k for k in captured["afterany"] if k.endswith("CPU_finalize.sh"))
    assert captured["afterany"][final] == [cpu]
    assert captured["publish_on_success"] == [final]
    import zipfile

    package = actual_prepare(
        root,
        captured["scripts"],
        afterany=captured["afterany"],
        extra_result_roots=captured["extra_result_roots"],
        publish_on_success=[final],
    )
    with zipfile.ZipFile(package["upload_archive"]) as archive:
        plan = json.loads(archive.read("submission_plan.json"))
        assert plan["afterany"][final] == [cpu]
        assert "_round_transfer.py" not in archive.read(cpu).decode("utf8")
        final_script = archive.read(final).decode("utf8")
        assert "phase_status" in final_script and "--finalize" in final_script


def test_relax_mc_cpu_groups_precede_phase_controller(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.tools.relax_mc_batch.dependencies import submit_analysis
    from phase_agent.tools.relax_mc_batch.phase_groups import prepare_groups
    import phase_agent.tools.relax_mc_batch.dependencies as dependencies

    tasks = [
        {
            "task_id": str(i),
            "task_key": str(i),
            "batch_id": "batch",
            "status": "pending",
            "result_path": str(tmp_path / "results" / str(i) / "result.json"),
        }
        for i in range(201)
    ]
    batch = {"batch_id": "batch", "job_id": "100", "task_ids": [t["task_id"] for t in tasks]}
    state = {"tasks": tasks, "relax_mc_remote": {"phase": 0, "phase_controllers": {"0": {}}}}
    manager = SimpleNamespace(boundary={}, data={"system_config": {}})
    manifest = {
        "bundle_id": "B",
        "phase_references": {},
        "stage_profiles": {"gpu": {"stages": ["relax_and_feature"], "slurm_options": {}}},
    }

    class IO:
        def __init__(self):
            self.calls = []

        def submit(self, path, *, dependency=None):
            self.calls.append((str(path), dependency))
            return str(200 + len(self.calls))

    io = IO()
    monkeypatch.setattr(dependencies, "checkpoint", lambda *args: None)
    result = submit_analysis(tmp_path, state, manager, manifest, io, [batch])
    assert len(io.calls) == 2
    assert io.calls[0][1] == "afterany:100"
    assert io.calls[1][1] == "afterany:201"
    assert result["analysis_job_id"] == "202"
    groups = state["relax_mc_remote"]["phase_controllers"]["0"]["phase_groups"]
    assert groups["group_count"] == 2
    assert "#SBATCH --array=0-1" in Path(groups["script_path"]).read_text()
    assert '"$SLURM_ARRAY_TASK_ID"' in Path(groups["script_path"]).read_text()
    assert prepare_groups(tmp_path, state, manager, manifest, [batch], 0)["group_count"] == 2
    state["relax_mc_remote"]["resume_attempt"] = {"plan_id": "retry1"}
    retry = prepare_groups(tmp_path, state, manager, manifest, [batch], 0)
    assert retry["spec_relative"] != groups["spec_relative"]
