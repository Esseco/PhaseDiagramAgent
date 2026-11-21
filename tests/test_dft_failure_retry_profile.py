import json

import pytest

from phase_agent.tools.workflows.dft_failure_retry import symmetry_retry_settings


def test_symmetry_retry_retains_physics_and_corrects_resources(tmp_path):
    (tmp_path / "workflow.json").write_text(json.dumps({"incar_settings": {"static": {"ENCUT": 680, "LDAUU": [0, 6.2, 0]}}, "kpoints_settings": {"static": {"reciprocal_density": 64}}}), encoding="utf-8")
    (tmp_path / "submit_gpu.sh").write_text("#!/bin/bash\n#SBATCH -n 2\n#SBATCH --ntasks-per-node=2\n#SBATCH --gres=gpu:1\npython3 workflow.py\n", encoding="utf-8")
    params, script = symmetry_retry_settings({"stage": "dft_single_point", "input_path": str(tmp_path / "task.json")})
    assert params["incar_settings"]["static"] == {"ENCUT": 680, "LDAUU": [0, 6.2, 0], "ISYM": -1, "SYMPREC": 1e-5}
    assert params["kpoints_settings"]["static"]["reciprocal_density"] == 64
    assert "#SBATCH --ntasks=1\n" in script
    assert "#SBATCH --ntasks-per-node=1\n" in script
    assert "#SBATCH --gres=gpu:1\n" in script


def test_symmetry_profile_refuses_non_dft():
    with pytest.raises(ValueError, match="requires_dft"):
        symmetry_retry_settings({"stage": "deep_search"})


@pytest.mark.parametrize("profile", ["vasp_symmetry_single_gpu", "mixing_parameters"])
def test_retry_profile_runs_through_retry_handler(tmp_path, monkeypatch, profile):
    from phase_agent.tools.workflows import create_workflow_handlers as handlers

    (tmp_path / "workflow.json").write_text(json.dumps({"incar_settings": {"static": {"ENCUT": 680}}}), encoding="utf-8")
    (tmp_path / "submit_gpu.sh").write_text(
        "#SBATCH -n 2\n#SBATCH --ntasks-per-node=2\n#SBATCH --gres=gpu:1\npython3 workflow.py\n",
        encoding="utf-8",
    )
    original = {"task_id": "T", "stage": "dft_single_point", "input_path": str(tmp_path / "task.json")}
    retry = {"task_key": "retry-T"}
    monkeypatch.setattr(handlers, "restart_failed_task", lambda *args, **kwargs: {"status": "prepared", "state": args[0], "task": retry})
    result = handlers._restart_failed_task(
        action={"target_ids": ["T"], "parameters": {"repair_profile": profile}},
        context={"event_state": {"tasks": [original]}},
    )
    incar = result["tasks"][0]["parameters"]["incar_settings"]["static"]
    assert incar["ENCUT"] == 680
    if profile == "vasp_symmetry_single_gpu":
        assert incar["ISYM"] == -1
    else:
        assert incar["AMIX"] == 0.2
    assert result["tasks"][0]["retry_repair_profile"] == profile


def test_input_retry_does_not_waive_other_pending_tasks():
    from types import SimpleNamespace
    from copy import deepcopy
    from phase_agent.tools.workflows.lifecycle_recovery import _failed_retry_input_only

    state = {"tasks": [{"task_id": "F", "stage": "dft_single_point", "status": "failed"}, {"task_id": "P", "stage": "dft_single_point", "status": "pending"}], "pending_dft_recovery_question": {"pending_task_ids": ["P"]}}
    before = deepcopy(state)
    frame = {"feedback": {"state": state}, "agent_client": SimpleNamespace(_failed_retry_target="F")}
    assert _failed_retry_input_only(frame)
    assert state == before
    frame["agent_client"]._failed_retry_target = "P"
    assert not _failed_retry_input_only(frame)
    frame["agent_client"] = None
    frame["human_feedback"] = {"decision": "approve"}
    state["pending_execution_policies"] = {"plan": {"agent_proposal": {"raw_action": {"action_type": "restart_failed_task", "target_ids": ["F"], "parameters": {"prepare_inputs_after_approval": True}}}}}
    assert _failed_retry_input_only(frame)
