from pathlib import Path
import hashlib
import pytest
from phase_agent.evidence.dft_extrapolation import candidate_extrapolation


def test_exact_frame_model_and_threshold_are_bound(tmp_path):
    frame = tmp_path / "a.vasp"
    frame.write_bytes(b"a")
    digest = hashlib.sha256(b"a").hexdigest()
    candidate = {"structure_path": str(frame), "structure_sha256": digest, "model_version": "m"}

    def task(score, threshold, failed, model="m"):
        return {
            "status": "completed",
            "model_version": model,
            "outputs": {
                "structure_checksum": digest,
                "extrapolation_monitor": {
                    "final_feature_difference": score,
                    "feature_difference_threshold": threshold,
                    "extrapolation_failure": failed,
                    "failures": [{"score": score}] if failed else [],
                },
            },
        }

    assert (
        candidate_extrapolation(candidate, {"tasks": [task(2.0, 3.0, False)]})["status"] == "normal"
    )
    assert (
        candidate_extrapolation(candidate, {"tasks": [task(2.0, 1.0, True)]})["status"] == "failed"
    )
    assert (
        candidate_extrapolation(candidate, {"tasks": [task(0.2, 1.0, False, "old")]})["status"]
        == "unknown"
    )
    assert (
        candidate_extrapolation(
            candidate, {"tasks": [task(0.2, 1.0, False), task(2.0, 1.0, True)]}
        )["status"]
        == "failed"
    )
    frame.write_bytes(b"other")
    assert (
        candidate_extrapolation(candidate, {"tasks": [task(0.2, 1.0, False)]})["status"]
        == "unknown"
    )


def test_mc_frame_requires_explicit_monitor(tmp_path):
    frame = tmp_path / "frame"
    frame.write_bytes(b"geometry")
    sha = hashlib.sha256(b"geometry").hexdigest()
    candidate = {"structure_path": str(frame), "structure_sha256": sha, "model_version": "m"}
    obs = {
        "structure_checksum": sha,
        "identity_scheme": "periodic_input_v1",
        "monitor_complete": True,
        "extrapolation_failure": False,
    }
    task = {
        "status": "completed",
        "model_version": "m",
        "outputs": {
            "relaxation_observations_complete": True,
            "actual_mc_steps": 0,
            "relaxation_observations": [obs],
        },
    }
    assert candidate_extrapolation(candidate, {"tasks": [task]})["status"] == "normal"
    obs["monitor_complete"] = False
    assert candidate_extrapolation(candidate, {"tasks": [task]})["status"] == "unknown"


def test_validator_blocks_failure_and_unknown_even_if_forced_uncertainty(tmp_path):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import (
        validate_dft_agent_decisions,
    )
    from phase_agent.configuration.defaults.default_dft_decision_config import (
        default_dft_decision_config,
    )

    frame = tmp_path / "frame"
    frame.write_bytes(b"a")
    sha = hashlib.sha256(b"a").hexdigest()
    metric = {
        "candidate_id": "c",
        "structure_path": str(frame),
        "structure_sha256": sha,
        "model_version": "m",
        "atom_count": 1,
        "f_std_max": 999,
    }
    state = {"active_model_version": "m", "tasks": []}
    proposal = {
        "source": "codex",
        "decisions": [{"candidate_id": "c", "action": "DFT_SINGLE_POINT", "reason": "check"}],
    }
    config = default_dft_decision_config()
    result = validate_dft_agent_decisions(
        proposal, [metric], state, config=config, config_version="v", remaining_budget=100000
    )
    assert not result["accepted"] and result["rejected"][0]["reason"] == "dft_extrapolation_unknown"
    state["tasks"] = [
        {
            "model_version": "m",
            "outputs": {
                "structure_checksum": sha,
                "extrapolation_monitor": {"extrapolation_failure": True},
            },
        }
    ]
    result = validate_dft_agent_decisions(
        proposal, [metric], state, config=config, config_version="v", remaining_budget=100000
    )
    assert not result["accepted"] and result["rejected"][0]["reason"] == "dft_extrapolation_failed"


def test_known_failed_input_is_excluded_from_training_without_changing_ledger(tmp_path):
    from copy import deepcopy
    from phase_agent.science.training.cumulative_training_records import cumulative_training_records

    failed = "a" * 64
    safe = "b" * 64
    state = {
        "tasks": [
            {
                "task_id": "r",
                "model_version": "m",
                "outputs": {
                    "structure_checksum": failed,
                    "extrapolation_monitor": {"extrapolation_failure": True},
                },
            },
            {
                "task_id": "d1",
                "stage": "dft_single_point",
                "model_version": "m",
                "input_structure_sha256": failed,
            },
            {
                "task_id": "d2",
                "stage": "dft_single_point",
                "model_version": "m",
                "input_structure_sha256": safe,
            },
        ],
        "dft_training_records": [
            {"task_id": "d1", "data_id": "one", "checks_passed": True},
            {"task_id": "d2", "data_id": "two", "checks_passed": True},
        ],
    }
    before = deepcopy(state)
    records = cumulative_training_records(state)
    assert [r["data_id"] for r in records] == ["two"]
    assert state == before


def test_known_failure_is_excluded_from_mixed_hull():
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from tests.test_latest_mixed_hull import record

    rows = [
        record("a0", "a", "aaa", "old", 0, -5),
        record("a1", "a", "aaa", "new", 0, -6),
        record("b0", "b", "bbb", "old", 1, -7),
    ]
    state = {
        "tasks": [
            {
                "task_id": "fail",
                "model_version": "new",
                "outputs": {
                    "structure_checksum": "aaa",
                    "extrapolation_monitor": {"extrapolation_failure": True},
                },
            }
        ],
        "upload_layout": {"model_rounds": {"old": 1, "new": 2}},
    }
    result = update_phase_diagram(
        rows, active_model_version="new", output_state=state, latest_per_structure=True
    )
    assert {r["record_id"] for r in result["diagrams"]["mlip"]["entries"]} == {"a0", "b0"}
    assert not result["mlip_by_version"]["new"]["entries"]


def test_preparing_approved_dft_rechecks_failure_without_writing(tmp_path):
    from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches

    path = tmp_path / "input.vasp"
    path.write_bytes(b"bad")
    sha = hashlib.sha256(b"bad").hexdigest()
    tasks = [
        {
            "task_id": "source",
            "model_version": "m",
            "outputs": {
                "structure_checksum": sha,
                "extrapolation_monitor": {"extrapolation_failure": True},
            },
        },
        {
            "task_id": "dft",
            "stage": "dft_single_point",
            "status": "pending",
            "model_version": "m",
            "structure_path": str(path),
            "input_structure_sha256": sha,
        },
    ]
    result = prepare_dft_upload_batches(
        action={"target_ids": ["dft"]},
        context={
            "event_state": {"tasks": tasks},
            "effective_config": {"upload_batches_directory": str(tmp_path / "delivery")},
        },
    )
    assert result["status"] == "rejected"
    assert result["reason"] == "dft_extrapolation_failed"
    assert not (tmp_path / "delivery").exists()


def test_failed_dft_input_is_excluded_from_dft_hull():
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from tests.test_latest_mixed_hull import record

    rows = [
        {**record("bad", "a", "aaa", "m", 0, -5), "energy_method": "dft", "source_task_id": "d1"},
        {**record("good", "b", "bbb", "m", 1, -7), "energy_method": "dft", "source_task_id": "d2"},
    ]
    state = {
        "tasks": [
            {
                "task_id": "r",
                "model_version": "m",
                "outputs": {
                    "structure_checksum": "aaa",
                    "extrapolation_monitor": {"extrapolation_failure": True},
                },
            },
            {
                "task_id": "d1",
                "stage": "dft_single_point",
                "model_version": "m",
                "input_structure_sha256": "aaa",
            },
            {
                "task_id": "d2",
                "stage": "dft_single_point",
                "model_version": "m",
                "input_structure_sha256": "bbb",
            },
        ]
    }
    result = update_phase_diagram(rows, output_state=state, include_dft=True)
    hull = result["diagrams"]["dft"]
    assert "bad" not in {r["record_id"] for r in hull["entries"]}
    assert any(
        r["record_id"] == "bad" and r["reason"] == "excluded_extrapolation_failure"
        for r in hull["rejected"]
    )


def test_refresh_cannot_use_relax_mc_threshold(tmp_path):
    frame = tmp_path / "frame"
    frame.write_bytes(b"a")
    sha = hashlib.sha256(b"a").hexdigest()
    task = {
        "status": "completed",
        "model_version": "m",
        "parameters": {"model_refresh_operation": "relax"},
        "outputs": {
            "structure_checksum": sha,
            "extrapolation_monitor": {
                "final_feature_difference": 2.0,
                "feature_difference_threshold": 3.0,
                "extrapolation_failure": False,
                "failures": [],
            },
        },
    }
    assert (
        candidate_extrapolation(
            {"structure_path": str(frame), "model_version": "m"}, {"tasks": [task]}
        )["status"]
        == "unknown"
    )
