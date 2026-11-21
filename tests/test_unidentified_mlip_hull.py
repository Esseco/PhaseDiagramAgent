import hashlib

import pytest

from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram


def fixture(tmp_path):
    from pymatgen.core import Lattice, Structure

    path = tmp_path / "verified.vasp"
    Structure(
        Lattice.cubic(10),
        ["Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.2, 0.2, 0.2],
            [0.7, 0.7, 0.7],
            [0.2, 0.2, 0.5],
            [0.2, 0.5, 0.2],
            [0.7, 0.7, 0.4],
            [0.7, 0.4, 0.7],
        ],
    ).to(filename=path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        "status": "unknown",
        "reason": "ValueError: unrecognized_phase:X",
        "structure_sha256": digest,
        "classification_scope": "scope",
    }
    composition = {"Na": 1, "Ni": 2, "O": 4}
    output = {
        "energy": -2,
        "energy_unit": "eV",
        "composition": composition,
        "structure_path": str(path),
        "structure_checksum": digest,
        "phase_identification": evidence,
        "optimizer_converged": True,
    }
    task = {
        "task_id": "x",
        "status": "completed",
        "stage": "relax_and_feature",
        "model_version": "m",
        "converged": True,
        "checks_passed": True,
        "outputs": output,
    }
    unknown = {
        "record_id": "x",
        "source_task_id": "x",
        "structure_id": "sx",
        "status": "pending_phase_identification",
        "phase_identification_status": "unknown",
        "phase_identification": evidence,
        "structure_path": str(path),
        "structure_sha256": digest,
        "composition": composition,
        "energy": -2,
        "energy_unit": "eV",
        "energy_method": "mlip",
        "source_version": "m",
        "model_version": "m",
        "checks_passed": True,
        "converged": True,
    }

    def endpoint(record_id, na, phase):
        endpoint_path = tmp_path / (record_id + ".vasp")
        species = (["Na"] if na else []) + ["Ni", "O", "O"]
        coords = ([[0, 0, 0]] if na else []) + [[0.2, 0.2, 0.2], [0.2, 0.2, 0.5], [0.2, 0.5, 0.2]]
        Structure(Lattice.cubic(10), species, coords).to(filename=endpoint_path)
        endpoint_digest = hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
        return {
            "record_id": record_id,
            "structure_id": record_id,
            "status": "completed",
            "structure_path": str(endpoint_path),
            "structure_sha256": endpoint_digest,
            "phase_identification_status": "identified",
            "phase": phase,
            "composition": {"Na": na, "Ni": 1, "O": 2},
            "energy": 0,
            "energy_unit": "eV",
            "energy_method": "mlip",
            "source_version": "m",
            "checks_passed": True,
        }

    state = {
        "active_model_version": "m",
        "tasks": [task],
        "confirmed_config": {
            "system": {
                "boundary": {
                    "P": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O1", "O3", "P3"]}
                }
            }
        },
    }
    return [endpoint("left", 0, "O1"), endpoint("right", 1, "O3"), unknown], state


def run(records, state):
    return update_phase_diagram(records, active_model_version="m", output_state=state)["diagrams"][
        "mlip"
    ]


def test_verified_unidentified_point_is_excluded_from_global_hull(tmp_path):
    records, state = fixture(tmp_path)
    diagram = run(records, state)
    assert diagram["status"] == "completed"
    assert {x["record_id"] for x in diagram["entries"]} == {"left", "right"}
    assert records[-1]["phase_identification_status"] == "unknown"


@pytest.mark.parametrize(
    "change",
    [
        "digest",
        "energy",
        "model",
        "extrapolation",
        "missing",
        "composition_geometry",
        "final_frame",
    ],
)
def test_inconsistent_or_excluded_unidentified_point_never_enters_hull(tmp_path, change):
    records, state = fixture(tmp_path)
    task = state["tasks"][0]
    if change == "digest":
        task["outputs"]["structure_checksum"] = "0" * 64
    elif change == "energy":
        task["outputs"]["energy"] = 123
    elif change == "model":
        task["model_version"] = "other"
    elif change == "extrapolation":
        task["outputs"]["extrapolation_monitor"] = {"extrapolation_failure": True}
    elif change == "composition_geometry":
        records[-1]["composition"] = task["outputs"]["composition"] = {"Na": 2, "Ni": 2, "O": 4}
    elif change == "final_frame":
        task["outputs"]["final_frame_valid"] = False
    else:
        state["tasks"].clear()
    assert all(x["record_id"] != "x" for x in run(records, state)["entries"])


def test_unidentified_plot_does_not_draw_an_unidentified_phase_hull():
    from matplotlib import pyplot as plt
    from Process_Phasediagram import plot_phase_diagram

    figure, axes = plot_phase_diagram(
        [0, 0.25, 0.5, 1],
        [0, 1, 2, 0],
        ["O1", "Unidentified", "Unidentified", "O3"],
        [True, False, False, True],
        phase_hulls=True,
    )
    try:
        assert "Unidentified" in [t.get_text() for t in axes.get_legend().get_texts()]
        assert all(line.get_linestyle() != "--" for line in axes.lines)
    finally:
        plt.close(figure)


def test_latest_hull_excludes_unidentified_point_and_exports(tmp_path):
    from phase_agent.analysis.phase.export_current_phase_diagram import export_current_phase_diagram
    records, state = fixture(tmp_path)
    state["upload_layout"] = {"model_rounds": {"m": 1}}
    diagram = update_phase_diagram(records, active_model_version="m", output_state=state, latest_per_structure=True)["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert all(x["record_id"] != "x" for x in diagram["entries"])
    csv = tmp_path / "existing.csv"
    csv.write_text("data", encoding="utf-8")
    diagram["csv_path"] = str(csv)
    state["phase_diagrams"] = {"mlip": diagram}
    actual, count, _ = export_current_phase_diagram(state)
    assert actual == csv and count == 2


def test_single_allowed_phase_does_not_accept_unidentified_label(tmp_path):
    records, state = fixture(tmp_path)
    state["confirmed_config"]["system"]["boundary"]["P"]["intermediate"] = ["O3"]
    assert all(x["record_id"] != "x" for x in run(records, state)["entries"])


def test_unknown_history_is_preserved_but_excluded_from_remote_reconstruction(tmp_path):
    from copy import deepcopy
    from phase_agent.tools.relax_mc_batch.plan import _latest_diagram_history
    records, state = fixture(tmp_path)
    state.update(phase_records=records, upload_layout={"model_rounds": {"m": 1}}, mlip_energy_policy="latest_valid_per_structure_v1")
    diagram = update_phase_diagram(records, active_model_version="m", output_state=state)["diagrams"]["mlip"]
    history = _latest_diagram_history(state, diagram, "m")
    assert {r["record_id"] for r in history} == {"left", "right"}
    remote = {**state, "tasks": [], "historical_phase_evidence_tasks": deepcopy(state["tasks"]), "historical_unidentified_required_ids": ["x"]}
    rebuilt = update_phase_diagram(history, active_model_version="m", output_state=remote)["diagrams"]["mlip"]
    assert rebuilt["status"] == "completed"
    assert all(x["record_id"] != "x" for x in rebuilt["entries"])
    assert state["phase_records"][-1]["record_id"] == "x"


def test_latest_unknown_refresh_does_not_revive_old_model_frame(tmp_path):
    from copy import deepcopy
    records, state = fixture(tmp_path)
    unknown = records[-1]
    old = deepcopy(unknown)
    old.update(record_id="old", status="completed", phase="P3",
               phase_identification_status="identified", model_version="m0", source_version="m0",
               source_task_id="old-task")
    task = state["tasks"][0]
    task.update(structure_id=unknown["structure_id"], input_structure_sha256=old["structure_sha256"],
                parameters={"model_refresh_operation": "predict"})
    state["upload_layout"] = {"model_rounds": {"m0": 1, "m": 2}}
    diagram = update_phase_diagram(records + [old], active_model_version="m", output_state=state,
                                   latest_per_structure=True)["diagrams"]["mlip"]
    assert diagram["status"] == "completed", diagram
    assert all(x["record_id"] not in {"x", "old"} for x in diagram["entries"])
    assert unknown["status"] == "pending_phase_identification"
