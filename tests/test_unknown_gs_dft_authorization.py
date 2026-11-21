from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from phase_agent.analysis.phase.phase_dft_candidates import bind_ground_state_frames
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.evidence.unknown_gs_dft import (
    is_authorized_unknown_gs_scope,
    verify_unknown_gs_frames,
)
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs
from tests.test_dft_ground_state_binding import fixture


def case(tmp_path):
    state, candidate, row = fixture(tmp_path)
    candidate = bind_ground_state_frames(state, [candidate])[0]
    digest = row["structure_sha256"]
    state.update(
        confirmed_config_version="c1",
        qbc_candidates=[candidate],
        tasks=[],
        exact_candidate_relax_authorization={"candidate_frame_sha256": {"s1": digest}},
        unknown_gs_dft_authorization={
            "status": "approved_for_preparation",
            "candidate_frame_sha256": {"s1": digest},
            "model_version": "m1",
            "config_version": "c1",
            "hull_version": "h1",
        },
    )
    decisions = [{"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "reason": "user validation"}]
    return state, decisions


def test_explicit_unknown_gs_exact_scope_previews_without_auxiliary_sampling(tmp_path):
    state, decisions = case(tmp_path)
    state["dft_audit_candidate_ids"] = ["other"]
    assert is_authorized_unknown_gs_scope(state, decisions)
    result, error = preview_dft_inputs(
        {"tool": "select_dft_candidates", "parameters": {"decisions": decisions}},
        state,
        default_layered_search_config(),
    )
    assert error is None
    assert result["candidate_ids"] == ["s1"] and result["single_point_count"] == 1
    assert result["ground_state_policy"] == "all_eligible_gs_v1"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "relax",
        "duplicate",
        "model",
        "config",
        "hull",
        "not_gs",
        "bad_hash",
        "failed",
        "normal",
        "already_dft",
    ],
)
def test_scope_rejects_widening_staleness_known_monitor_and_existing_dft(tmp_path, mutation):
    state, decisions = case(tmp_path)
    row = state["qbc_candidates"][0]
    if mutation == "missing":
        decisions = []
    if mutation == "extra":
        decisions.append({"candidate_id": "s2", "action": "DFT_SINGLE_POINT"})
    if mutation == "relax":
        decisions[0]["action"] = "DFT_RELAX"
    if mutation == "duplicate":
        decisions.append(decisions[0])
    if mutation == "model":
        state["active_model_version"] = "other"
    if mutation == "config":
        state["confirmed_config_version"] = "other"
    if mutation == "hull":
        state["phase_diagrams"]["mlip"]["version"] = "other"
    if mutation == "not_gs":
        state["phase_diagrams"]["mlip"]["entries"][0]["is_stable"] = False
    if mutation == "bad_hash":
        Path(row["structure_path"]).write_text("changed")
    if mutation in {"failed", "normal"}:
        state["tasks"].append(
            {
                "status": "completed",
                "model_version": "m1",
                "outputs": {
                    "structure_checksum": row["structure_sha256"],
                    "extrapolation_monitor": {
                        "extrapolation_failure": mutation == "failed",
                        "final_feature_difference": 0.2,
                        "failures": [{"score": 2}] if mutation == "failed" else [],
                    },
                },
            }
        )
    if mutation == "already_dft":
        state["tasks"].append(
            {
                "status": "pending",
                "stage": "dft_single_point",
                "input_structure_sha256": row["structure_sha256"],
            }
        )
    assert not is_authorized_unknown_gs_scope(state, decisions)


def test_scope_requires_explicit_preexisting_frame_authorization(tmp_path):
    state, decisions = case(tmp_path)
    state["exact_candidate_relax_authorization"] = {}
    with pytest.raises(ValueError, match="unknown_gs_candidate_binding"):
        verify_unknown_gs_frames(
            state, state["unknown_gs_dft_authorization"]["candidate_frame_sha256"]
        )


def test_codex_preflight_only_exempts_exact_independent_scope():
    from phase_agent.runtime.codex.context import dft_audit_preflight_errors

    context = {
        "independent_audit_candidate_ids": ["other"],
        "unknown_gs_dft_authorization": {
            "status": "approved_for_preparation",
            "candidate_frame_sha256": {"s1": "hash"},
        },
    }
    action = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "s1", "action": "DFT_SINGLE_POINT"}]},
    }
    assert dft_audit_preflight_errors(action, context) == []
    action["parameters"]["decisions"][0]["action"] = "DFT_RELAX"
    assert dft_audit_preflight_errors(action, context) == ["independent_dft_audit_required"]


def test_agent_authorization_records_only_explicit_bound_scope(tmp_path, monkeypatch):
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime import unknown_gs_dft_authorization as mod

    state, decisions = case(tmp_path)
    frames = state.pop("unknown_gs_dft_authorization")["candidate_frame_sha256"]
    saves = []
    monkeypatch.setattr(mod, "read_json", lambda *a: state)
    monkeypatch.setattr(mod, "read_identity", lambda *a, **k: "version")
    monkeypatch.setattr(
        mod,
        "_save_runner_state",
        lambda s, p, **kw: saves.append(deepcopy(s["unknown_gs_dft_authorization"])),
    )
    control = SimpleNamespace(
        state_path=str(tmp_path / "state.json"), chat_handler=SimpleNamespace(lock=RLock())
    )
    kwargs = dict(
        candidate_frame_sha256=frames,
        expected_model="m1",
        expected_config="c1",
        expected_hull="h1",
        user_instruction="Explicitly allow these unknown GS for independent DFT single points",
    )
    receipt = mod.authorize(control, **kwargs)
    assert receipt == {"status": "authorized", "candidate_count": 1, "submit_authorized": False}
    assert saves[0]["candidate_frame_sha256"] == frames
    assert is_authorized_unknown_gs_scope(state, decisions)
    assert mod.authorize(control, **kwargs)["status"] == "already_authorized"
    assert len(saves) == 1
    with pytest.raises(ValueError, match="explicit_unknown_gs"):
        mod.authorize(control, **{**kwargs, "user_instruction": ""})
    with pytest.raises(ValueError, match="unknown_gs_context_changed"):
        mod.authorize(control, **{**kwargs, "expected_model": "other"})


def test_retained_old_energy_and_current_model_monitor_are_separately_bound(tmp_path):
    state, decisions = case(tmp_path)
    row = state["qbc_candidates"][0]
    row["model_version"] = "old"
    row["energy_source_version"] = "old"
    diagram = state["phase_diagrams"]["mlip"]
    diagram.update(mixed_model_versions=True, energy_policy="latest_valid_per_structure_v1")
    diagram["entries"][0]["source_version"] = "old"
    state["tasks"].append(
        {
            "status": "completed",
            "model_version": "old",
            "outputs": {
                "structure_checksum": row["structure_sha256"],
                "extrapolation_monitor": {"extrapolation_failure": True},
            },
        }
    )
    assert is_authorized_unknown_gs_scope(state, decisions)
    state["tasks"][0]["model_version"] = "m1"
    assert not is_authorized_unknown_gs_scope(state, decisions)
    state["tasks"] = []
    row["energy_source_version"] = "different"
    assert not is_authorized_unknown_gs_scope(state, decisions)


def task_case(tmp_path):
    state, decisions = case(tmp_path)
    row = state["qbc_candidates"][0]
    task = {
        "task_id": "d1",
        "object_id": "s1",
        "structure_id": "s1",
        "stage": "dft_single_point",
        "status": "pending",
        "structure_path": row["structure_path"],
        "input_structure_sha256": row["structure_sha256"],
        "model_version": "m1",
        "config_version": "c1",
        "parent_decision_id": "approved-plan",
        "scientific_parent_task_key": "selection",
    }
    action = {
        "tool": "select_dft_candidates",
        "task_key": "selection",
        "target_ids": ["s1"],
        "parameters": {"decisions": decisions},
    }
    state["tasks"] = [task]
    state["action_records"] = [
        {
            "record_id": "approved-plan",
            "status": "rejected",
            "human_feedback": {"decision": "approve"},
            "agent_proposal": {"raw_action": action},
        }
    ]
    return state, task, action


def test_approved_registered_tasks_reuse_exact_scope_after_interrupted_input(tmp_path):
    from phase_agent.evidence.unknown_gs_dft import is_authorized_unknown_gs_tasks

    state, task, action = task_case(tmp_path)
    assert is_authorized_unknown_gs_tasks(state, [task])
    assert is_authorized_unknown_gs_tasks(
        {**state, "action_records": []},
        [task],
        approval_record_id="approved-plan",
        approved_action=action,
    )
    altered = deepcopy(state)
    altered["action_records"][0]["human_feedback"]["decision"] = "reject"
    assert not is_authorized_unknown_gs_tasks(altered, [task])
    altered = deepcopy(task)
    altered["stage"] = "dft_relax"
    assert not is_authorized_unknown_gs_tasks(state, [altered])
    altered = deepcopy(task)
    altered["slurm_batch_id"] = "submitted"
    assert not is_authorized_unknown_gs_tasks(state, [altered])
    altered = deepcopy(task)
    altered["input_structure_sha256"] = "0" * 64
    assert not is_authorized_unknown_gs_tasks(state, [altered])
    assert not is_authorized_unknown_gs_tasks(state, [task, task])


def test_input_gate_allows_only_approved_unknown_scope(tmp_path, monkeypatch):
    from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches
    from phase_agent.configuration.schema import python_environments

    state, task, _ = task_case(tmp_path)

    def reached(*a, **kw):
        raise RuntimeError("reached after monitor gate")

    monkeypatch.setattr(python_environments, "python_environment", reached)
    context = {
        "event_state": state,
        "effective_config": {"upload_batches_directory": str(tmp_path)},
    }
    with pytest.raises(RuntimeError, match="reached after monitor gate"):
        prepare_dft_upload_batches(action={"target_ids": ["d1"]}, context=context)
    state["action_records"][0]["human_feedback"]["decision"] = "reject"
    result = prepare_dft_upload_batches(action={"target_ids": ["d1"]}, context=context)
    assert result["status"] == "rejected" and result["reason"] == "dft_extrapolation_unknown"


def test_post_recovery_allows_only_exact_registered_authorized_input_tasks(tmp_path, monkeypatch):
    from phase_agent.policies import recovery_transition as policy

    state, task, _ = task_case(tmp_path)
    monkeypatch.setattr(policy, "_post_recovery_base_tools", lambda s: {"select_dft_candidates"})
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_dft_inputs",
        "parameters": {"mode": "dft_inputs"},
        "target_ids": ["d1"],
    }
    assert "prepare_local_batch_files" in policy.post_recovery_tools(state)
    assert policy.exact_recovery_scope_errors(action, state) == []
    changed = deepcopy(action)
    changed["target_ids"] = ["d1", "unapproved"]
    assert policy.exact_recovery_scope_errors(changed, state)
    changed = deepcopy(action)
    changed["parameters"]["mode"] = "relax_inputs"
    assert policy.exact_recovery_scope_errors(changed, state)
    state["action_records"][0]["human_feedback"]["decision"] = "reject"
    assert policy.exact_recovery_scope_errors(action, state)
