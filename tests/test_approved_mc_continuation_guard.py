from copy import deepcopy
from phase_agent.agents.strategic.codex_team import _approved_mc_continuation


def test_only_approved_exact_continuation():
    session = {"status": "submitted", "team_id": "team"}
    action = {
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "mc_inputs"},
        "target_ids": ["m1"],
    }
    original = {
        "tool": "prepare_local_batch_files",
        "parameters": {
            "mode": "relax_mc_inputs",
            "relax_mc_preview": {"bundle_id": "b", "checksum": "sha"},
        },
        "_decision_team": {"team_id": "team"},
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "phase_diagrams": {"mlip": {"version": "h"}},
        "tasks": [
            {
                "task_id": "m1",
                "stage": "deep_search",
                "status": "pending",
                "relax_mc_bundle_id": "b",
            }
        ],
        "relax_mc_batches": {
            "b": {
                "plan_checksum": "sha",
                "model_version": "m",
                "config_version": "c",
                "threshold_transition": {
                    "mc_first_wave": {
                        "status": "allocated_not_submitted",
                        "task_ids": ["m1"],
                        "hull_version": "h",
                        "original_action_record_id": "r",
                    }
                },
            }
        },
        "action_records": [
            {"record_id": "r", "human_feedback": {"decision": "approve"}, "final_action": original}
        ],
    }
    assert _approved_mc_continuation(state, session, action)
    for alteration in ("targets", "approved", "model", "hull", "team", "prepared"):
        s = deepcopy(state)
        a = deepcopy(action)
        t = deepcopy(session)
        if alteration == "targets":
            a["target_ids"].append("other")
        if alteration == "approved":
            s["action_records"][0]["human_feedback"]["decision"] = "reject"
        if alteration == "model":
            s["active_model_version"] = "different"
        if alteration == "hull":
            s["phase_diagrams"]["mlip"]["version"] = "different"
        if alteration == "team":
            t["team_id"] = "different"
        if alteration == "prepared":
            s["tasks"][0]["slurm_batch_id"] = "existing"
        assert not _approved_mc_continuation(s, t, a), alteration
