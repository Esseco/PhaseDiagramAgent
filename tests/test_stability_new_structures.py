"""Held-out chemistry: real generation, ordering, classification, import and persistence; fake energies only."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.science.structures.generate_branch_structure import generate_branch_structure
from phase_agent.science.structures.identify_branch import identify_branch_parameters
from phase_agent.science.structures.initialize_branch_structures import initialize_branch_structures
from phase_agent.tools.relax_mc_batch.plan import attach_preview
from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
from phase_agent.tools.relax_mc_batch.run import run_batch
from phase_agent.tools.relax_mc_batch.results import import_results
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.relax_mc_batch.files import read, write
from phase_agent.graphs.approval_graph import durable_approval
from phase_agent.runtime.project_diagnostics import inspect_project
from tests.test_relax_mc_batch import fake_executor


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("h")


def new_project(root, ratio, frames=1):
    root.mkdir(exist_ok=True)
    mother = Structure.from_spacegroup(
        166, Lattice.hexagonal(3.0, 15.5), ["Na", "Ni", "O"], [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.24]]
    )
    matrices = [[[2, i, 0], [0, 1, 0], [0, 0, 1]] for i in range(frames)]
    boundary = {"P": ["O3"], "H": {"O3": matrices}, "TM_ratio": ratio}
    manager = PhaseDataManager(boundary)
    references = {"O3": mother}
    for matrix in matrices:
        for x in [0, "1/2", 1]:
            structure = generate_branch_structure(
                boundary, phase="O3", H=matrix, x=x, phase_references=references, seed=812
            )
            actual = identify_branch_parameters(
                structure, boundary, phase_references=references, phase_hint="O3"
            )
            assert Fraction(actual["x"]) == Fraction(x)
            bid = manager.add_branch(
                P="O3", H=matrix, x=x, T=actual["T"], composition=actual["composition"]
            )
            branch = manager.data["branches"][bid]
            candidates = initialize_branch_structures(
                [branch], boundary, references, initial_states_per_branch=1, seed=813
            )
            assert candidates
            candidate = candidates[0]
            path = root / (bid + ".vasp")
            candidate["structure"].to(filename=path, fmt="poscar")
            sid = manager.add_structure(
                branch_id=bid,
                arrangement=candidate["arrangement"],
                composition=candidate["structure"].composition.as_dict(),
                source_path=path,
                metadata={
                    key: candidate[key]
                    for key in (
                        "initialization_method",
                        "electrostatic_energy",
                        "electrostatic_rank",
                        "electrostatic_charge_scheme",
                    )
                },
            )
            if candidate["full_na_structure"] is not None:
                full = root / (bid + "-full.vasp")
                candidate["full_na_structure"].to(filename=full, fmt="poscar")
                manager.data["structures"][sid]["full_na_structure_path"] = str(full)
                branch["full_na_structure_path"] = str(full)
    reference = root / "mother.vasp"
    mother.to(filename=reference, fmt="poscar")
    model = root / "fake.model"
    model.write_text("fake weights never loaded")
    config = default_layered_search_config(boundary=boundary)
    config["mlip"].update(name="m1", version="m1", model_path=str(model))
    config["upload_batches_directory"] = str(root / "u")
    config["bohb"]["relax_structures_per_branch"] = 1
    config.pop("ledger_path", None)
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": "heldout-v1",
        "dedup_gate": {"status": "ready"},
        "tasks": [],
    }
    context = {
        "effective_config": config,
        "manager": manager,
        "event_state": state,
        "config_version": "heldout-v1",
        "phase_references": {"O3": str(reference)},
    }
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_relax_mc_batch",
        "task_key": "new-chemistry",
        "target_ids": list(manager.data["branches"]),
        "parameters": {"mode": "relax_mc_inputs"},
        "reason": "held-out regression",
        "budget": 0,
    }
    return manager, state, config, context, action


@pytest.mark.parametrize(
    "ratio,frames", [({"Ni": 1}, 5), ({"Ti": 1, "V": 1}, 5), ({"Ni": 1, "Ti": 1, "V": 1}, 5)]
)
def test_heldout_structure_closed_loop_without_mocking_recognition(
    tmp_path, monkeypatch, ratio, frames
):
    manager, state, config, context, action = new_project(tmp_path, ratio, frames)
    monkeypatch.setattr(
        "phase_agent.tools.remote.handoff_naming.batch_label",
        lambda *a, **k: "epoch0_RelaxMC_Batch01",
    )
    action = attach_preview(action, state, context, config)
    envelope = {
        "invocation_id": "heldout",
        "proposal_hash": "immutable-new-geometry",
        "config_version": "heldout-v1",
        "revision": 0,
    }
    assert durable_approval(tmp_path / "state.json", envelope)["status"] == "awaiting_approval"
    assert not (tmp_path / "u").exists()
    assert (
        durable_approval(tmp_path / "state.json", envelope, "approve")["status"]
        == "decision_delivered"
    )
    prepared = prepare_batch(action=action, context=context)
    assert prepared["status"] == "prepared" and not prepared["submitted"]
    from phase_agent.analysis.phase import ensure_phase_identification as phase_module
    from phase_agent.analysis.phase import build_composition_hull_entries as hull_module

    real_identify, real_uniformity = (
        phase_module.identify_result_phase,
        hull_module.check_na_layer_uniformity,
    )
    execute, calls = fake_executor(monkeypatch)
    # Reuse only fake energy/worker receipts. Real phase detection and Na layer gates stay enabled.
    monkeypatch.setattr(phase_module, "identify_result_phase", real_identify)
    monkeypatch.setattr(hull_module, "check_na_layer_uniformity", real_uniformity)

    def fake_physics(batch, state, manager):
        results = execute(batch, state, manager)
        for result in results:
            if result["stage"] == "deep_search":
                # A lower synthetic MC energy must describe a different exact geometry.
                geometry_path = Path(result["outputs"]["structure_path"])
                geometry = Structure.from_file(geometry_path)
                sodium = next(i for i, site in enumerate(geometry) if site.specie.symbol == "Na")
                geometry.translate_sites(
                    [sodium],
                    [0.0001 * (result.get("segment_index", 0) + 1), 0, 0],
                    frac_coords=True,
                    to_unit_cell=True,
                )
                geometry.to(filename=geometry_path, fmt="poscar")
            result["outputs"].update(
                optimizer_converged=True,
                force_max_ev_per_angstrom=0.01,
                fmax_target_ev_per_angstrom=0.05,
            )
            task = next(t for t in state["tasks"] if t["task_id"] == result["task_id"])
            result_path = Path(task["result_path"])
            if result_path.name != "result.json":
                result_path = result_path.parent / "result.json"
            write(result_path, result)
            marker_path = result_path.with_name("task.finished.json")
            marker = read(marker_path)
            marker["result_checksum"] = file_checksum(result_path)
            write(marker_path, marker)
        return results

    outcome = run_batch(prepared["directory"], submit=True, batch_executor=fake_physics)
    assert outcome["status"] == "completed", outcome
    call_count = len(calls)
    repeated = run_batch(prepared["directory"], submit=True, batch_executor=fake_physics)
    assert len(calls) == call_count and not repeated["submitted"]
    returned = Path(prepared["expected_results_archive"])
    returned.parent.mkdir(exist_ok=True)
    shutil.copy2(outcome["archive"], returned)
    imported = import_results(
        action={"parameters": {"bundle_id": prepared["bundle_id"]}},
        context={**context, "event_state": prepared["state"]},
    )
    assert imported["status"] == "completed", imported
    new = imported["state"]
    assert not new.get("scientific_feedback_errors")
    assert new["phase_diagrams"]["mlip"]["status"] == "completed"
    assert any(t["stage"] == "deep_search" for t in new["tasks"])
    if frames == 5:
        assert {index for stage, index, _ in calls if stage == "deep_search"} == {0, 1, 2}
    na0 = [
        t
        for t in new["tasks"]
        if t["stage"] == "relax_and_feature"
        and not (t.get("outputs") or {}).get("composition", {}).get("Na")
    ]
    assert na0 and all((t.get("outputs") or {}).get("actual_phase") == "O3" for t in na0)
    assert all(
        t.get("phase_identification_status") == "identified"
        for t in new["tasks"]
        if t["status"] == "completed"
    )
    assert not any(
        "fake" in (t.get("outputs") or {}).get("phase_identification", {}).get("method", "")
        for t in new["tasks"]
    )
    before = deepcopy(manager.data)
    replay = import_results(
        action={"parameters": {"bundle_id": prepared["bundle_id"]}},
        context={**context, "event_state": new},
    )
    assert replay["status"] == "already_imported" and manager.data == before
    manager.save(tmp_path / "ledger.json")
    assert PhaseDataManager.load(tmp_path / "ledger.json").data == manager.data
    write(tmp_path / "state.json", new)
    report = inspect_project(tmp_path / "state.json")
    assert report["sections"]["state"]["available"]
    (tmp_path / "heldout-audit.json").write_text(
        json.dumps(
            {
                "chemistry": ratio,
                "branches": len(manager.data["branches"]),
                "tasks": len(new["tasks"]),
                "fake_compute_calls": len(calls),
                "real_generation": True,
                "real_identification": True,
                "duplicate_import_skipped": True,
                "diagnostic_readable": True,
            }
        ),
        encoding="utf-8",
    )


def test_heldout_na_free_and_multi_phase_geometry_recognition():
    from phase_agent.analysis.phase.identify_result_phase import identify_result_phase

    mother = Structure.from_spacegroup(
        166, Lattice.hexagonal(3.0, 15.5), ["Na", "V", "O"], [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.24]]
    )
    boundary = {
        "P": ["O1", "O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"V": 1},
    }
    manager = PhaseDataManager(boundary)
    source = {
        "task_id": "new-V",
        "stage": "deep_search",
        "status": "completed",
        "outputs": {"structure": mother.as_dict()},
    }
    identified = identify_result_phase(source, manager)
    assert identified["actual_phase"] == "O3"
    assert identified["outputs"]["phase_identification"]["method"] not in {
        "unique_configured_phase",
        "Na_free_O1_default",
    }
    mother.remove_species(["Na"])
    source["outputs"]["structure"] = mother.as_dict()
    identified = identify_result_phase(source, manager)
    assert identified["actual_phase"] == "O1"
    assert identified["outputs"]["phase_identification"]["method"] == "Na_free_O1_default"
