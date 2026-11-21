from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from phase_agent.analysis.phase.identify_result_phase import identify_result_phase
from phase_agent.analysis.phase.ensure_phase_identification import ensure_phase_identification


def _manager():
    return SimpleNamespace(
        data={"branches": {"B": {"P": "O3"}}, "structures": {}, "system_config": {}}
    )


def _result(path, stage="relax_and_feature"):
    return {
        "task_id": stage,
        "stage": stage,
        "status": "completed",
        "branch_id": "B",
        "outputs": {
            "structure_path": str(path),
            "single_point_completed": stage == "dft_single_point",
        },
    }


def test_portable_rules_reuse_and_changed_rules_invalidate(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    manager = _manager()
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "P3", "method": "test"},
        ) as detector,
    ):
        first = identify_result_phase(_result(path), manager)
        assert first["outputs"]["phase_identification"]["portable_classification_scope"]
        moved = deepcopy(first)
        moved["outputs"]["phase_identification"]["classification_scope"] = "remote-host-scope"
        assert identify_result_phase(moved, manager)["actual_phase"] == "P3"
        assert detector.call_count == 1
        manager.data["system_config"]["species"] = {"mobile": ["Na", "Li"]}
        identify_result_phase(moved, manager)
        assert detector.call_count == 2


def test_single_point_inherits_only_identical_bound_structure(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    manager = _manager()
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "P3", "method": "test"},
        ) as detector,
    ):
        source = identify_result_phase(_result(path), manager)
        single = _result(path, "dft_single_point")
        result, _ = ensure_phase_identification({"tasks": [single, source]}, manager)
        assert result["tasks"][0]["actual_phase"] == "P3"
        assert detector.call_count == 1
        changed = tmp_path / "changed.vasp"
        changed.write_text("different")
        ensure_phase_identification(
            {"tasks": [_result(changed, "dft_single_point"), source]}, manager
        )
        assert detector.call_count == 2


def test_unknown_evidence_inherits_without_fabricating_phase(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    manager = _manager()
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "X"},
        ),
        patch(
            "phase_agent.science.structures.identify_branch.identify_phase",
            return_value={"phase": "X"},
        ) as detector,
    ):
        source = identify_result_phase(_result(path), manager)
        assert source["phase_identification_status"] == "unknown"
        result, _ = ensure_phase_identification(
            {"tasks": [_result(path, "dft_single_point"), source]}, manager
        )
        assert result["tasks"][0]["phase_identification_status"] == "unknown"
        assert result["tasks"][0]["actual_phase"] is None
        assert detector.call_count == 2  # source creation and source retry; not the single point


def test_reference_rules_are_portable_and_content_bound(tmp_path):
    from phase_agent.analysis.phase.identify_result_phase import _portable_classification_scope

    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    local = tmp_path / "local.vasp"
    remote = tmp_path / "remote.vasp"
    structure.to(filename=local, fmt="poscar")
    structure.to(filename=remote, fmt="poscar")
    manager = _manager()
    one = _portable_classification_scope(manager, {"O3": str(local)})
    assert one is not None
    assert one == _portable_classification_scope(manager, {"O3": str(remote)})
    changed = structure.copy()
    changed.translate_sites([0], [0.1, 0, 0])
    changed.to(filename=remote, fmt="poscar")
    assert one != _portable_classification_scope(manager, {"O3": str(remote)})


def test_legacy_result_without_binding_still_classifies(tmp_path):
    path = tmp_path / "old.vasp"
    path.write_text("legacy")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result = _result(path)
    result["outputs"]["phase_identification"] = {"status": "identified", "phase": "O3"}
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "P3", "method": "test"},
        ) as detector,
    ):
        assert identify_result_phase(result, _manager())["actual_phase"] == "P3"
        assert detector.call_count == 1


def test_changed_algorithm_reloads_structure_when_old_file_cache_matches(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "P3", "method": "test"},
        ) as detector,
    ):
        prior = identify_result_phase(_result(path), _manager())
        with patch(
            "phase_agent.analysis.phase.identify_result_phase._portable_classification_scope",
            return_value="new-algorithm",
        ):
            result = identify_result_phase(prior, _manager())
        assert detector.call_count == 2
        assert result["actual_phase"] == "P3"
        assert (
            result["outputs"]["phase_identification"]["portable_classification_scope"]
            == "new-algorithm"
        )


def test_verified_remote_unknown_is_not_reidentified_locally(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    manager = _manager()
    with (
        patch("pymatgen.core.Structure.from_file", return_value=structure),
        patch(
            "phase_agent.science.structures.identify_layered_phase_fast.identify_layered_phase_fast",
            return_value={"phase": "X"},
        ),
        patch(
            "phase_agent.science.structures.identify_branch.identify_phase",
            return_value={"phase": "X"},
        ) as detector,
    ):
        remote = identify_result_phase(_result(path), manager)
        remote["outputs"]["phase_identification"]["origin"] = "remote_cpu"
        local = identify_result_phase(remote, manager)
        assert local["phase_identification_status"] == "unknown"
        assert detector.call_count == 1
