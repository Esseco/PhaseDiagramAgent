from hashlib import sha256
from pymatgen.core import Lattice, Structure
import pytest
from phase_agent.analysis.phase.phase_dft_candidates import bind_ground_state_frames


def fixture(tmp_path):
    def make(name, shift):
        p = tmp_path / name
        Structure(
            Lattice.cubic(4),
            ["Na", "Ni", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75 + shift]],
        ).to(filename=str(p), fmt="poscar")
        return str(p), sha256(p.read_bytes()).hexdigest()

    old, oldsha = make("old.vasp", 0)
    new, newsha = make("new.vasp", 0.1)
    row = {
        "record_id": "gs1",
        "structure_id": "s1",
        "structure_path": old,
        "structure_sha256": oldsha,
        "is_stable": True,
        "phase": "O3",
        "phase_identification_status": "identified",
        "composition": {"Na": 1, "Ni": 1, "O": 2},
        "x_Na_per_O2": 1,
        "ehull": 0.0,
        "ehull_unit": "eV/atom",
        "energy_per_O2": -1.0,
    }
    state = {
        "active_model_version": "m1",
        "phase_diagrams": {
            "mlip": {
                "status": "completed",
                "model_version": "m1",
                "version": "h1",
                "entries": [row],
            }
        },
    }
    candidate = {
        "candidate_id": "s1",
        "structure_id": "s1",
        "branch_id": "b1",
        "structure_path": new,
        "structure_sha256": newsha,
        "predicted_Ehull": 0.2,
        "qbc": {"status": "completed", "f_std_max": 1},
    }
    return state, candidate, row


def test_binds_gs_frame_and_does_not_borrow_qbc(tmp_path):
    state, candidate, row = fixture(tmp_path)
    result = bind_ground_state_frames(state, [candidate])[0]
    assert result["structure_path"] == row["structure_path"]
    assert result["structure_sha256"] == row["structure_sha256"]
    assert result["predicted_Ehull"] == 0
    assert result["qbc"]["status"] == "not_available_exact_current_geometry"
    assert result["branch_id"] == "b1"


def test_inequivalent_gs_same_id_is_rejected(tmp_path):
    state, candidate, row = fixture(tmp_path)
    other = dict(
        row,
        record_id="gs2",
        structure_path=candidate["structure_path"],
        structure_sha256=candidate["structure_sha256"],
    )
    state["phase_diagrams"]["mlip"]["entries"].append(other)
    result = bind_ground_state_frames(state, [candidate])
    frames = [r for r in result if r.get("ground_state_record_ids")]
    assert len(frames) == 2
    assert len({r["candidate_id"] for r in frames}) == 2
    assert {r["structure_sha256"] for r in frames} == {row["structure_sha256"], candidate["structure_sha256"]}
    assert all(r["structure_id"] == "s1" for r in frames)
    other_frame = next(r for r in frames if r["structure_sha256"] == row["structure_sha256"])
    assert other_frame["qbc"]["status"] == "not_available_exact_current_geometry"


def test_corrupt_gs_is_rejected(tmp_path):
    state, candidate, row = fixture(tmp_path)
    row["structure_sha256"] = "bad"
    with pytest.raises(ValueError, match="ground_state_frame_invalid"):
        bind_ground_state_frames(state, [candidate])


def test_selection_requires_ground_state_and_actual_frame(tmp_path):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import (
        _ground_state_selection_errors,
    )

    state, candidate, row = fixture(tmp_path)
    state["qbc_candidates"] = [candidate]
    assert _ground_state_selection_errors(state, {"other"}) == [
        "ground_state_selection_required:s1"
    ]
    assert _ground_state_selection_errors(state, {"s1"}) == [
        "ground_state_frame_binding_required:s1"
    ]
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    assert _ground_state_selection_errors(state, {"s1"}) == []


def test_direct_recovery_gs_exemption_requires_exact_verified_batch(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import _is_direct_recovery_selection
    import phase_agent.tools.workflows.direct_extrapolation_sampling as direct

    frame = tmp_path / "direct.vasp"
    frame.write_text("recovered frame", encoding="utf8")
    digest = sha256(frame.read_bytes()).hexdigest()
    selected = [{"candidate_id": "direct-1", "structure_sha256": digest}]
    receipt = {"source_digest": "source", "model_version": "bad", "selected": selected}
    state = {"extrapolation_recovery_cycle": {"status": "awaiting_reviewed_dft_labels", "scope_id": "refresh", "invalid_model_version": "bad", "direct_source_digest": "source", "selected": selected}, "direct_sampling_results": {"refresh": receipt}, "qbc_candidates": [{"candidate_id": "direct-1", "structure_path": str(frame), "structure_sha256": digest}]}
    monkeypatch.setattr(direct, "valid_direct_receipt", lambda value: value is receipt)
    assert _is_direct_recovery_selection(state, {"direct-1"})
    assert not _is_direct_recovery_selection(state, {"direct-1", "extra"})
    assert not _is_direct_recovery_selection(state, set())
    frame.write_text("changed frame", encoding="utf8")
    assert not _is_direct_recovery_selection(state, {"direct-1"})
    frame.write_text("recovered frame", encoding="utf8")
    receipt["source_digest"] = "different"
    assert not _is_direct_recovery_selection(state, {"direct-1"})
    receipt["source_digest"] = "source"
    state["extrapolation_recovery_cycle"]["status"] = "completed"
    assert not _is_direct_recovery_selection(state, {"direct-1"})


def test_direct_exemption_does_not_relax_ordinary_gs_rule(tmp_path, monkeypatch):
    import phase_agent.tools.workflows.validate_dft_agent_decisions as validator
    state, candidate, _ = fixture(tmp_path)
    state["qbc_candidates"] = [candidate]
    monkeypatch.setattr(validator, "_is_direct_recovery_selection", lambda state, ids: False)
    assert validator._ground_state_selection_errors(state, {"other"}) == ["ground_state_selection_required:s1"]


def test_dft_validator_skips_gs_only_for_verified_direct_recovery(tmp_path, monkeypatch):
    from phase_agent.evidence import dft_extrapolation
    # This unit test isolates the GS/direct guard; extrapolation is verified separately.
    monkeypatch.setattr(dft_extrapolation, "candidate_extrapolation", lambda *args: {"status": "normal"})
    from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
    from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
    import phase_agent.tools.workflows.direct_extrapolation_sampling as direct

    state, candidate, _ = fixture(tmp_path)
    frame = tmp_path / "direct.vasp"
    frame.write_text("frame", encoding="utf8")
    digest = sha256(frame.read_bytes()).hexdigest()
    selected = [{"candidate_id": "direct", "structure_sha256": digest}]
    receipt = {"source_digest": "digest", "model_version": "bad", "selected": selected}
    state["qbc_candidates"] = [candidate, {"candidate_id": "direct", "structure_path": str(frame), "structure_sha256": digest}]
    state["direct_sampling_results"] = {"refresh": receipt}
    state["extrapolation_recovery_cycle"] = {"status": "awaiting_reviewed_dft_labels", "scope_id": "refresh", "invalid_model_version": "bad", "direct_source_digest": "digest", "selected": selected}
    monkeypatch.setattr(direct, "valid_direct_receipt", lambda value: value is receipt)
    metric = {"candidate_id": "direct", "branch_id": "B", "atom_count": 10}
    proposal = {"decisions": [{"candidate_id": "direct", "action": "DFT_SINGLE_POINT", "reason": "DIRECT recovery"}], "source": "llm_agent"}
    def run():
        return validate_dft_agent_decisions(proposal, [metric], state, config=default_dft_decision_config(), config_version="c1", remaining_budget=2000)
    assert run()["valid"]
    state["extrapolation_recovery_cycle"]["status"] = "completed"
    assert "ground_state_selection_required:s1" in run()["errors"]


def test_ground_state_requires_selection_or_explicit_reasoned_deferral(tmp_path):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import _ground_state_selection_errors
    state, candidate, _ = fixture(tmp_path)
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    assert _ground_state_selection_errors(state, {"other"}) == ["ground_state_selection_required:s1"]
    assert _ground_state_selection_errors(state, {"other"}, {"s1"}) == []
    assert _ground_state_selection_errors(state, {"s1"}, {"s1"}) == []


def test_ground_state_reasoned_deferral_must_be_valid_decision(tmp_path, monkeypatch):
    from phase_agent.evidence import dft_extrapolation
    # A synthetic normal candidate lets this test exercise the GS deferral guard.
    monkeypatch.setattr(dft_extrapolation, "candidate_extrapolation", lambda *args: {"status": "normal"})
    from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
    from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
    state, candidate, _ = fixture(tmp_path)
    state["qbc_candidates"] = bind_ground_state_frames(state, [candidate])
    metrics = [{"candidate_id": "other", "branch_id": "b2", "atom_count": 10}, {"candidate_id": "s1", "branch_id": "b1", "atom_count": 4}]
    def run(reason):
        proposal = {"decisions": [{"candidate_id": "other", "action": "DFT_SINGLE_POINT", "reason": "validate"}, {"candidate_id": "s1", "action": "DEFER", "reason": reason}], "source": "codex"}
        return validate_dft_agent_decisions(proposal, metrics, state, config=default_dft_decision_config(), config_version="c1", remaining_budget=2000)
    assert "ground_state_selection_required:s1" in run(None)["errors"]
    assert "ground_state_selection_required:s1" not in run("user discarded this failure")["errors"]


def test_selection_candidate_provider_includes_exact_bound_ground_state_frames(tmp_path):
    from phase_agent.tools.workflows.create_dft_selection_handler import _state_candidates
    state, candidate, row = fixture(tmp_path)
    state["qbc_candidates"] = []
    from copy import deepcopy
    other = deepcopy(row)
    other.update(structure_id="s2", record_id="gs2")
    state["phase_diagrams"]["mlip"]["entries"].append(other)
    candidates = _state_candidates(state, None, {})
    assert {q["candidate_id"] for q in candidates} == {"s1", "s2"}
    assert all(q["ground_state_frame_sha256"] == row["structure_sha256"] for q in candidates)


def test_ground_state_uses_existing_exact_frame_committee(tmp_path):
    state, candidate, row = fixture(tmp_path)
    qbc = {"status": "completed", "energy_per_atom_std": 0.003, "member_count": 4}
    state["tasks"] = [
        {"task_id": "wrong", "status": "completed", "model_version": "old", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": qbc}},
        {"task_id": "exact", "status": "completed", "model_version": "m1", "checks_passed": True, "outputs": {"structure_checksum": row["structure_sha256"], "qbc": qbc}},
    ]
    bound = bind_ground_state_frames(state, [candidate])[0]
    assert bound["qbc"] == qbc
    assert bound["qbc_frame_sha256"] == row["structure_sha256"]
    assert bound["qbc_model_version"] == "m1"
    assert bound["qbc_source_task_id"] == "exact"


def test_ground_state_does_not_bind_conflicting_committee_results(tmp_path):
    state, candidate, row = fixture(tmp_path)
    state["tasks"] = [
        {"task_id": "a", "status": "completed", "model_version": "m1", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": {"status": "completed", "energy_per_atom_std": 0.001}}},
        {"task_id": "b", "status": "completed", "model_version": "m1", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": {"status": "completed", "energy_per_atom_std": 0.002}}},
    ]
    bound = bind_ground_state_frames(state, [candidate])[0]
    assert bound["qbc"]["status"] == "not_available_exact_current_geometry"


def test_ground_state_does_not_reuse_extrapolation_failure_qbc(tmp_path):
    state, candidate, row = fixture(tmp_path)
    state["tasks"] = [{"task_id": "failed", "status": "completed", "model_version": "m1", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": {"status": "completed"}, "extrapolation_monitor": {"extrapolation_failure": True}}}]
    assert bind_ground_state_frames(state, [candidate])[0]["qbc"]["status"] == "not_available_exact_current_geometry"


def test_ground_state_accepts_roundoff_in_matching_committee_results(tmp_path):
    state, candidate, row = fixture(tmp_path)
    state["tasks"] = [
        {"task_id": "a", "status": "completed", "model_version": "m1", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": {"status": "completed", "force_rms_disagreement": 0.023574681649401595}}},
        {"task_id": "b", "status": "completed", "model_version": "m1", "outputs": {"structure_checksum": row["structure_sha256"], "qbc": {"status": "completed", "force_rms_disagreement": 0.023574681649401578}}},
    ]
    assert bind_ground_state_frames(state, [candidate])[0]["qbc"]["status"] == "completed"
