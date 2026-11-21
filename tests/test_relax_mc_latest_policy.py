"""Synthetic cross-epoch hull and Na/vacancy endpoint regression."""

import pytest

from phase_agent.analysis.phase.mc_snapshot import mc_model_snapshot
from phase_agent.science.mc.schedule_tiered_mc import _mc_candidate_exclusion_reason
from phase_agent.tools.relax_mc_batch.plan import _latest_diagram_history


def sample():
    records = [
        {
            "record_id": "old",
            "structure_id": "s0",
            "structure_sha256": "a" * 64,
            "model_version": "m0",
            "energy": -2.0,
            "status": "completed",
            "checks_passed": True,
        },
        {
            "record_id": "new",
            "structure_id": "s1",
            "structure_sha256": "b" * 64,
            "model_version": "m1",
            "energy": -3.0,
            "status": "completed",
            "checks_passed": True,
        },
    ]
    entries = [
        {
            "record_id": r["record_id"],
            "phase": "O3",
            "structure_id": r["structure_id"],
            "structure_sha256": r["structure_sha256"],
            "source_version": r["model_version"],
            "normalized_total_energy": r["energy"],
        }
        for r in records
    ]
    diagram = {
        "status": "completed",
        "version": "v1",
        "model_version": "m1",
        "energy_policy": "latest_valid_per_structure_v1",
        "mixed_model_versions": True,
        "source_model_versions": ["m0", "m1"],
        "entries": entries,
    }
    state = {
        "active_model_version": "m1",
        "mlip_energy_policy": "latest_valid_per_structure_v1",
        "phase_records": records,
        "phase_diagrams": {"mlip": diagram},
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2}},
    }
    return state, diagram


def test_latest_history_freezes_every_provenanced_record():
    state, diagram = sample()
    history = _latest_diagram_history(state, diagram, "m1")
    assert [r["record_id"] for r in history] == ["old", "new"]
    assert {r["model_version"] for r in history} == {"m0", "m1"}
    assert mc_model_snapshot(state, "m1") is diagram


@pytest.mark.parametrize("mutation", ["missing", "energy", "geometry", "model", "epoch"])
def test_latest_history_rejects_changed_or_unverifiable_record(mutation):
    state, diagram = sample()
    if mutation == "missing":
        state["phase_records"].pop(0)
    if mutation == "energy":
        state["phase_records"][0]["energy"] = -99
    if mutation == "geometry":
        state["phase_records"][0]["structure_sha256"] = "c" * 64
    if mutation == "model":
        state["phase_records"][0]["model_version"] = "m1"
    if mutation == "epoch":
        del state["upload_layout"]["model_rounds"]["m0"]
    with pytest.raises(ValueError):
        _latest_diagram_history(state, diagram, "m1")


def test_na_v_mc_excludes_both_endpoints_without_excluding_relax():
    assert _mc_candidate_exclusion_reason({"x": "0"}) == "no_mobile_ion_sites"
    assert (
        _mc_candidate_exclusion_reason({"x": "1", "mc_no_vacancies": True}) == "no_vacant_ion_sites"
    )
    assert _mc_candidate_exclusion_reason({"x": "1"}) is None
    assert _mc_candidate_exclusion_reason({"x": "1/2"}) is None
