"""Actual project geometries, private approvals and fake HPC physics only."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import shutil

import pytest

from tests.test_relax_mc_batch import fixture, fake_executor
from phase_agent.tools.relax_mc_batch.plan import attach_preview
from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
from phase_agent.tools.relax_mc_batch.run import run_batch
from phase_agent.tools.relax_mc_batch.results import recover_available
from phase_agent.tools.relax_mc_batch import upload_driver
from phase_agent.graphs.approval_graph import durable_approval


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("z")


@pytest.mark.parametrize("project", ["O3-1ele-NaCoO2", "O3-1ele-NaCrO2", "O3-2ele-NaFeMn"])
@pytest.mark.real_project
def test_actual_project_geometries_approval_input_output_analysis(tmp_path, monkeypatch, project):
    source = Path("E:/0-PhaseDiagram") / project / "workflow_state/ledgers/phase_data.json"
    if not source.is_file():
        pytest.skip("read-only project fixture unavailable")
    original = source.read_bytes()
    ledger = json.loads(original)
    manager, state, config, context, action = fixture(tmp_path / "l", count=3)
    config["upload_batches_directory"] = str(tmp_path)
    monkeypatch.setattr(
        "phase_agent.tools.remote.handoff_naming.batch_label",
        lambda *a, **k: "epoch0_RelaxMC_Batch01",
    )
    branches, structures = {}, {}
    available = [
        b
        for b in ledger["branches"].values()
        if b["P"] == "O3"
        and Path(b.get("full_na_structure_path", "")).is_file()
        and any(
            Path(ledger["structures"][sid]["source_path"]).is_file() for sid in b["structure_ids"]
        )
    ]
    ordered = sorted(
        available, key=lambda b: (Fraction(str(b["x"])), b.get("det_H", 999), b["branch_id"])
    )
    # Continuation requires at least ten branches; exercise it without reducing that rule.
    assert len(ordered) >= 13, project
    selected = [ordered[i * (len(ordered) - 1) // 12] for i in range(13)]
    assert len({Fraction(str(b["x"])) for b in selected}) >= 3
    for index, selected_branch in enumerate(selected):
        branch = deepcopy(selected_branch)
        sid = next(
            sid
            for sid in branch["structure_ids"]
            if Path(ledger["structures"][sid]["source_path"]).is_file()
        )
        structure = deepcopy(ledger["structures"][sid])
        local = tmp_path / "g" / str(index)
        local.mkdir(parents=True)
        shutil.copy2(structure["source_path"], local / "s.vasp")
        shutil.copy2(branch["full_na_structure_path"], local / "full.vasp")
        branch.update(structure_ids=[sid], full_na_structure_path=str(local / "full.vasp"))
        structure.update(
            source_path=str(local / "s.vasp"),
            full_na_structure_path=str(local / "full.vasp"),
            stage_history={stage: [] for stage in manager.stages},
        )
        branches[branch["branch_id"]] = branch
        structures[sid] = structure
    manager.data.update(branches=branches, structures=structures)
    context["phase_references"] = {"O3": str(local / "full.vasp")}
    action["target_ids"] = list(branches)
    action = attach_preview(action, state, context, config)
    envelope = {
        "invocation_id": "copy-probe",
        "proposal_hash": "frozen-geometry",
        "config_version": "v1",
        "revision": 0,
    }
    path = tmp_path / "state.json"
    assert durable_approval(path, envelope)["status"] == "awaiting_approval"
    assert not (tmp_path / "relax_mc_batches").exists()
    assert durable_approval(path, envelope, "approve")["status"] == "decision_delivered"
    prepared = prepare_batch(action=action, context=context)
    assert prepared["status"] == "prepared", prepared
    folder = Path(prepared["upload_folder"])
    assert folder.name.endswith("_input")
    assert {p.name for p in folder.iterdir()} == {"inputs.zip", "submit_all.sh", "run_upload.py"}
    executor, calls = fake_executor(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "fake-project-copy")

    def compute(*args, **kwargs):
        outcome = run_batch(kwargs["cwd"], submit=True, batch_executor=executor)
        assert outcome["status"] == "completed", outcome
        return 0

    monkeypatch.setattr(upload_driver.subprocess, "call", compute)
    assert upload_driver.main(folder) == 0
    output = folder.with_name(folder.name[:-6] + "_output")
    assert {p.name for p in output.iterdir()} == {"results.zip", "diagnostics.zip"}
    (tmp_path / "local_state.json").write_text(json.dumps(prepared["state"]), encoding="utf-8")
    (tmp_path / "local_context.json").write_text(
        json.dumps(
            {
                "manager": manager.data,
                "config": config,
                "phase_references": context["phase_references"],
            }
        ),
        encoding="utf-8",
    )
    recovered = recover_available(prepared["state"], context)
    record = recovered["relax_mc_batches"][prepared["bundle_id"]]
    assert record["status"] == "completed", record.get("return_error")
    assert recovered["qbc_candidates"]
    assert {stage for stage, _, _ in calls} == {"relax_and_feature", "deep_search"}
    assert {index for stage, index, _ in calls if stage == "deep_search"} == {0, 1, 2}
    assert source.read_bytes() == original
