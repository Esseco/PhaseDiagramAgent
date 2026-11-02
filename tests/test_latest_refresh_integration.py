import hashlib
from ase import Atoms
from ase.io import write
from tests.test_model_refresh_workflow import setup
from phase_agent.tools.workflows.model_refresh_state import refresh_preview


def test_explicit_latest_policy_uses_each_latest_source_and_keeps_versions(tmp_path):
    state, session, context, _ = setup(tmp_path)
    p = tmp_path / "prior.vasp"
    write(
        p,
        Atoms(
            "Na2FeO2",
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        format="vasp",
    )
    manager = context["manager"]
    manager.data["structures"]["t"] = {
        "branch_id": "bt",
        "source_path": str(p),
        "composition": {"Na": 2, "Fe": 1, "O": 2},
    }
    sfile = manager.data["structures"]["s"]["source_path"]

    def record(rid, sid, path, model, comp, energy):
        return {
            "record_id": rid,
            "structure_id": sid,
            "structure_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "structure_path": str(path),
            "model_version": model,
            "source_version": model,
            "energy_method": "mlip",
            "energy_unit": "eV",
            "energy": energy,
            "status": "completed",
            "composition": comp,
            "phase": "O3",
        }

    state["phase_records"] = [
        record("s", "s", sfile, "old", {"Na": 1, "Fe": 1, "O": 2}, -10),
        record("t", "t", p, "older", {"Na": 2, "Fe": 1, "O": 2}, -12),
    ]
    state["upload_layout"] = {"model_rounds": {"older": 1, "old": 2, "new": 3}}
    cfg = {
        **context["effective_config"],
        "model_refresh": {"energy_policy": "latest_valid_per_structure_v1"},
    }
    plan = refresh_preview(state, manager, cfg)
    assert plan["energy_policy"] == "latest_valid_per_structure_v1"
    assert plan["mixed_model_versions"] is True
    assert {r["structure_id"]: r["energy_source_version"] for r in plan["candidates"]} == {
        "s": "old",
        "t": "older",
    }
    assert all(r["ehull"] is not None for r in plan["candidates"])


def test_unknown_model_epoch_blocks_preview(tmp_path):
    state, _, context, _ = setup(tmp_path)
    path = context["manager"].data["structures"]["s"]["source_path"]
    state["phase_records"] = [
        {
            "record_id": "r",
            "structure_id": "s",
            "structure_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "structure_path": path,
            "model_version": "unknown",
            "energy_method": "mlip",
            "energy_unit": "eV",
            "energy": -1,
            "status": "completed",
            "composition": {"Na": 1, "Fe": 1, "O": 2},
            "phase": "O3",
        }
    ]
    state["upload_layout"] = {"model_rounds": {"old": 2, "new": 3}}
    cfg = {
        **context["effective_config"],
        "model_refresh": {"energy_policy": "latest_valid_per_structure_v1"},
    }
    import pytest

    with pytest.raises(ValueError, match="谱系未解决"):
        refresh_preview(state, context["manager"], cfg)


def test_policy_name_and_shape_are_validated():
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.configuration.schema.validate_search_config import validate_search_config

    cfg = default_layered_search_config()
    cfg["model_refresh"] = {"energy_policy": "latest_valid_per_structure_v1"}
    assert "model_refresh.energy_policy_invalid" not in validate_search_config(cfg)["conflicts"]
    cfg["model_refresh"] = {"energy_policy": "typo"}
    assert "model_refresh.energy_policy_invalid" in validate_search_config(cfg)["conflicts"]
    cfg["model_refresh"] = "invalid"
    assert "model_refresh_must_be_object" in validate_search_config(cfg)["conflicts"]


def test_latest_policy_is_explicit_scientific_constraint():
    from phase_agent.configuration.session.classify_config_permission import classify_config_permission
    from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config

    config = {"model_refresh": {"energy_policy": "latest_valid_per_structure_v1"}}
    assert classify_config_permission("model_refresh.energy_policy", config) == "hard_constraint"
    effective = build_effective_run_config({"status": "confirmed", "confirmed_snapshot": {"config": config, "config_version": "c"}})
    assert effective["model_refresh"]["energy_policy"] == "latest_valid_per_structure_v1"
