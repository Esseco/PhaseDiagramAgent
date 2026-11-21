from copy import deepcopy
from hashlib import sha256

from phase_agent.analysis.phase.phase_dft_candidates import phase_review_candidates
from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates
from tests.test_phase_quality import sample, add


def files(state, tmp_path):
    for row in state["phase_diagrams"]["mlip"]["entries"]:
        path = tmp_path / (row["record_id"] + ".vasp")
        from pymatgen.core import Structure, Lattice

        x = float(row["composition"].get("Na", 0))
        count = round(2 * x)
        species = ["Co", "Co", "O", "O", "O", "O"] + ["Na"] * count
        coords = [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.2, 0, 0],
            [0, 0.2, 0],
            [0.7, 0.5, 0.5],
            [0.5, 0.7, 0.5],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
        ][: len(species)]
        structure = Structure(Lattice.cubic(5 + 0.01 * len(row["record_id"])), species, coords)
        structure.to(filename=str(path), fmt="poscar")
        # Distinguish equal-size fixtures without changing the physical composition.
        path.write_text(row["record_id"] + "\n" + "\n".join(path.read_text().splitlines()[1:]))
        row["composition"] = structure.composition.as_dict()
        row.update(
            structure_path=str(path),
            structure_sha256=sha256(path.read_bytes()).hexdigest(),
            ehull=0.0,
            ehull_unit="eV/atom",
        )


def test_review_pool_exposes_missing_phases_without_borrowing_old_QBC(tmp_path):
    state = sample()
    files(state, tmp_path)
    entry = state["phase_diagrams"]["mlip"]["entries"][1]
    state["tasks"] = [
        {
            "task_id": "old",
            "model_version": "old",
            "status": "completed",
            "outputs": {
                "structure_checksum": entry["structure_sha256"],
                "qbc": {"status": "completed", "f_std_max": 99},
            },
        }
    ]
    saved = [{"candidate_id": "saved", "model_version": "old"}]
    before = deepcopy((state, saved))
    result = phase_review_candidates(state, saved)
    assert (state, saved) == before and result[0]["candidate_id"] == "saved"
    row = next(row for row in result if row["candidate_id"] == entry["structure_id"])
    assert row["qbc"]["status"].startswith("not_available")
    assert row["dft_sampling_evidence"]["status"] == "available"
    assert row["dft_sampling_evidence"]["quality_priority_eligible"]
    assert post_mc_candidates(state, "m1")
    changed = deepcopy(state)
    changed["phase_diagrams"]["mlip"]["model_version"] = "stale"
    assert phase_review_candidates(changed, saved) == saved


def test_changed_files_and_outside_review_scope_are_not_supplemented(tmp_path):
    state = sample()
    add(state, "far", 10, x=0.25)
    files(state, tmp_path)
    report = diagnose_phase_diagram(state)
    entry = state["phase_diagrams"]["mlip"]["entries"][1]
    from pathlib import Path

    Path(entry["structure_path"]).write_text("changed")
    result = phase_review_candidates(state, [], report=report)
    assert entry["structure_id"] not in {row["candidate_id"] for row in result}
    assert "far" not in {row["candidate_id"] for row in result}


def test_exact_far_snapshot_binding_distinguishes_unscored_D_from_missing_evidence():
    from phase_agent.analysis.phase.phase_quality_reports import candidate_quality_evidence

    state = sample()
    for i in range(8):
        add(state, "far" + str(i), 10 + 0.001 * i, x=0.25)
    report = diagnose_phase_diagram(state)
    assert not any("far0" in row["structure_ids"] for row in report["structures"])
    result = candidate_quality_evidence(
        state, [{"candidate_id": "far", "structure_sha256": "far0"}], report=report
    )[0]
    assert result["dft_sampling_evidence"]["status"] == "available"
    assert result["dft_sampling_evidence"]["D_score"] is None
    assert result["dft_sampling_evidence"]["D_status"] == "large_cluster_not_scored"
    assert not result["dft_sampling_evidence"]["quality_priority_eligible"]
    assert result["phase_quality_signal"]["status"] == "unavailable"


def test_revision_receives_high_D_breadth_reference(monkeypatch):
    from phase_agent.agents.strategic.revise_tool_proposal import revise_tool_proposal
    from phase_agent.analysis.phase.phase_dft_sampling import dft_sampling_reference

    captured = []

    def request(client, payload):
        captured.append(payload)
        return {
            "tool": "select_dft_candidates",
            "parameters": {"decisions": []},
            "reason": "重新核查",
        }

    monkeypatch.setattr(
        "phase_agent.graphs.decision_team.backend.request_decision_action", request
    )
    revise_tool_proposal(
        {"raw_action": {"tool": "select_dft_candidates"}},
        "按新D/H扩大覆盖",
        state={},
        allowed_tools=["select_dft_candidates"],
        agent_client=object(),
        config={"dft": {"selection": {"single_point_cost_per_round": 5000}}},
    )
    policy = captured[0]["decision_context"]["dft_selection_policy"]
    assert policy["reference_framework"] == dft_sampling_reference()
    assert policy["single_point_cost_per_round"] is None
    assert policy["max_tasks_per_round"] == 100
    assert "独立低能簇" in str(policy["reference_framework"]["adaptation"])


def test_related_upper_reference_is_available_without_automatic_priority(tmp_path):
    state = sample()
    add(state, "reference", 0.021)
    files(state, tmp_path)
    result = phase_review_candidates(state, [])
    reference = next(row for row in result if row["candidate_id"] == "s2")
    assert reference["candidate_origin"] == "D_reference"
    assert not reference["dft_sampling_evidence"]["quality_priority_eligible"]


def test_initial_refresh_qbc_is_bound_to_exact_frame_and_model(tmp_path):
    state = sample()
    files(state, tmp_path)
    entry = state["phase_diagrams"]["mlip"]["entries"][1]
    digest = entry["structure_sha256"]
    initial = {"status": "completed", "single_point_completed": True,
               "structure_checksum": digest, "mlip_version": "m1",
               "qbc": {"status": "completed", "energy_per_atom_std": 0.005, "member_count": 4}}
    state["tasks"] = [{"task_id": "refresh", "model_version": "m1", "status": "completed",
        "input_structure_sha256": digest, "outputs": {"structure_checksum": "other-frame",
        "refresh_force_followup": {"input_structure_sha256": digest, "relaxed": True,
                                   "initial_prediction": initial}}}]
    result = phase_review_candidates(state, [])
    row = next(row for row in result if row["candidate_id"] == entry["structure_id"])
    assert row["qbc"]["energy_per_atom_std"] == 0.005
    assert row["qbc_frame_sha256"] == digest and row["qbc_model_version"] == "m1"
    assert row["source_task_id"] == "refresh"
    from phase_agent.evidence.dft_extrapolation import candidate_extrapolation

    assert candidate_extrapolation(row, state)["status"] == "unknown"
    state["tasks"][0]["outputs"]["refresh_force_followup"]["initial_prediction"]["mlip_version"] = "other"
    row = next(row for row in phase_review_candidates(state, []) if row["candidate_id"] == entry["structure_id"])
    assert row["qbc"]["status"] == "not_available_exact_current_geometry"
    state["tasks"][0]["outputs"]["refresh_force_followup"]["initial_prediction"]["mlip_version"] = "m1"
    state["tasks"][0]["outputs"]["refresh_force_followup"]["input_structure_sha256"] = "wrong"
    row = next(row for row in phase_review_candidates(state, []) if row["candidate_id"] == entry["structure_id"])
    assert row["qbc"]["status"] == "not_available_exact_current_geometry"



def test_saved_qbc_requires_current_task_and_exact_frame():
    from phase_agent.analysis.phase.phase_dft_candidates import _verify_candidate_qbc

    qbc = {"status": "completed", "member_count": 4, "energy_per_atom_std": 0.01}
    rows = [
        {"candidate_id": "a", "structure_sha256": "frame-a", "qbc": dict(qbc)},
        {"candidate_id": "b", "structure_sha256": "frame-b", "qbc": dict(qbc)},
        {"candidate_id": "c", "structure_sha256": "frame-c", "qbc": dict(qbc)},
    ]
    state = {"tasks": [
        {"task_id": "good", "status": "completed", "model_version": "m1",
         "outputs": {"structure_checksum": "frame-a", "qbc": dict(qbc)}},
        {"task_id": "old", "status": "completed", "model_version": "m0",
         "outputs": {"structure_checksum": "frame-b", "qbc": dict(qbc)}},
        {"task_id": "different", "status": "completed", "model_version": "m1",
         "outputs": {"structure_checksum": "frame-other", "qbc": dict(qbc)}},
    ]}
    _verify_candidate_qbc(state, rows, "m1")
    assert rows[0]["qbc_source_task_id"] == "good"
    assert rows[0]["qbc_frame_sha256"] == "frame-a"
    assert rows[0]["qbc_model_version"] == "m1"
    assert all(row["qbc"]["status"] == "not_available_exact_current_geometry" for row in rows[1:])
