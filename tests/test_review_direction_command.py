import json
from copy import deepcopy
from phase_agent.tools.local.review_direction_command import review_direction_command
from phase_agent.tools.local.recover_remote_training import inspect_training_results
from phase_agent.graphs.direction_review_graph import direction_checkpoint
from tests.test_remote_training_recovery import returned_job


def setup(tmp_path):
    job, root = returned_job(tmp_path)
    result = inspect_training_results(job)
    proposal = {"direction": "supplement_dft", "training_fingerprint": result["fingerprint"], "reason": "coverage"}
    job["training_handoff"] = {"stage": "awaiting_direction_approval", "direction_proposal": proposal}
    state = {"active_model_version": "base", "remote_finetune_jobs": {"j": job}}
    path = tmp_path / "state.json"
    direction_checkpoint(path, proposal)
    return state, path, root


def test_restart_approval_routes_without_computation(tmp_path):
    state, path, root = setup(tmp_path)
    restarted = json.loads(json.dumps(state))
    before = deepcopy(restarted)
    result = review_direction_command("同意", restarted, path)
    assert restarted == before
    assert result["state"]["active_model_version"] == "base"
    assert result["state"]["remote_finetune_jobs"]["j"]["training_handoff"]["stage"] == "validation_prerequisites_required"
    assert review_direction_command("同意", result["state"], path) is None
    assert list(root.glob("GPU*")) == []


def test_changed_results_block_old_approval(tmp_path):
    state, path, root = setup(tmp_path)
    models = json.loads((root / "models.json").read_text())
    models[0]["sha256"] = "b"*64
    (root / "models.json").write_text(json.dumps(models))
    result = review_direction_command("同意", state, path)
    assert result["state"] is state
    assert "证据已变化" in result["reason"]


def test_rejection_and_ambiguous_approval(tmp_path):
    state, path, _ = setup(tmp_path)
    rejected = review_direction_command("拒绝", state, path)
    assert rejected["state"]["remote_finetune_jobs"]["j"]["training_handoff"]["stage"] == "direction_rejected"
    state["pending_execution_policies"] = {"other": {}}
    assert "多个审批对象" in review_direction_command("同意", state, path)["reason"]
