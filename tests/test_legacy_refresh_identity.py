import hashlib
import io
import json

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.agents.reliability.extrapolation_round import assess_recovered_round
from phase_agent.agents.reliability.legacy_refresh_identity import (
    verified_legacy_refresh_observation,
)


def make_legacy(tmp_path):
    inputs = tmp_path / "inputs" / "task"
    results = tmp_path / "results" / "task"
    inputs.mkdir(parents=True)
    results.mkdir(parents=True)
    frame = io.StringIO()
    write(
        frame,
        Atoms("NaO", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True),
        format="vasp",
    )
    raw = frame.getvalue().encode()
    (inputs / "initial.vasp").write_bytes(raw)
    keys = {
        "task_id": "legacy",
        "task_key": "key",
        "task_checksum": "checksum",
        "model_version": "m",
        "config_version": "c",
    }
    record = {**keys, "input_structure_sha256": hashlib.sha256(raw).hexdigest()}
    (inputs / "task.json").write_text(json.dumps(record), encoding="utf8")
    monitor = {
        "final_feature_difference": 1.5,
        "failures": [{"score": 1.5}],
        "extrapolation_failure": True,
    }
    result = {**keys, "status": "completed", "outputs": {"extrapolation_monitor": monitor}}
    result_bytes = json.dumps(result).encode()
    (results / "result.json").write_bytes(result_bytes)
    marker = {
        **keys,
        "status": "completed",
        "result_checksum": hashlib.sha256(result_bytes).hexdigest(),
    }
    (results / "task.finished.json").write_text(json.dumps(marker), encoding="utf8")
    return {
        **record,
        "structure_id": "legacy-parent",
        "stage": "relax_and_feature",
        "status": "completed",
        "input_path": str(inputs / "task.json"),
        "result_path": str(results / "evaluated.vasp"),
        "parameters": {"model_refresh_operation": "relax"},
        "outputs": {"extrapolation_monitor": monitor},
    }


def test_verified_legacy_identity_counts_in_round(tmp_path):
    legacy = make_legacy(tmp_path)
    evidence = verified_legacy_refresh_observation(legacy)
    assert evidence["identity_scheme"] == "periodic_input_v1"
    assert evidence["extrapolation_failure"] is True
    modern = []
    for i in range(9):
        monitor = {"final_feature_difference": 0.2, "failures": [], "extrapolation_failure": False}
        modern.append(
            {
                "task_id": str(i),
                "structure_id": str(i),
                "model_version": "m",
                "stage": "relax_and_feature",
                "status": "completed",
                "outputs": {
                    "extrapolation_monitor": monitor,
                    "relaxation_observation": {
                        "structure_id": str(i),
                        "identity_scheme": "periodic_input_v1",
                        "monitor_complete": True,
                        "extrapolation_failure": False,
                    },
                },
            }
        )
    outcome = assess_recovered_round(
        modern + [legacy], model_version="m", scope_complete=True, round_kind="epoch_refresh"
    )
    assert outcome["structure_count"] == 10
    assert outcome["failure_count"] == 1
    assert outcome["direct_sampling_required"] is True


@pytest.mark.parametrize("alter", ["input", "marker", "result", "state_monitor", "model"])
def test_legacy_identity_rejects_unbound_or_changed_evidence(tmp_path, alter):
    task = make_legacy(tmp_path)
    if alter == "input":
        (tmp_path / "inputs" / "task" / "initial.vasp").write_text("changed")
    elif alter == "marker":
        (tmp_path / "results" / "task" / "task.finished.json").write_text("{}")
    elif alter == "result":
        (tmp_path / "results" / "task" / "result.json").write_text("{}")
    elif alter == "state_monitor":
        task["outputs"]["extrapolation_monitor"] = {"extrapolation_failure": False}
    else:
        task["model_version"] = "different"
    assert verified_legacy_refresh_observation(task) is None
    outcome = assess_recovered_round(
        [task], model_version=task["model_version"], scope_complete=True, round_kind="epoch_refresh"
    )
    assert outcome["direct_sampling_required"] is None
