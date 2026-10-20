from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.tools.policy.model_generation import (
    new_model_generation_facts,
    new_model_search_errors,
)
from phase_agent.tools.policy.branch_conditions import branch_entry_errors


def switched():
    return {
        "active_model_version": "new",
        "model_refresh": {
            "status": "completed",
            "new_model_version": "new",
            "old_model_version": "old",
            "plan": {"checksum": "refresh1"},
        },
        "generation_history": [
            {"model_version": "old", "registered_ids": ["s0"], "new_branch_ids": ["old_branch"]}
        ],
        "tasks": [],
    }


def generated():
    s = switched()
    s["generation_history"].append(
        {
            "model_version": "new",
            "registered_ids": ["s1"],
            "new_branch_ids": ["b1"],
            "refresh_plan_checksum": "refresh1",
        }
    )
    return s


@pytest.mark.parametrize(
    "tool,mode",
    [
        ("select_dft_candidates", None),
        ("allocate_mc_bohb", None),
        ("prepare_local_batch_files", "relax_mc_inputs"),
        ("prepare_local_batch_files", "relax_inputs"),
        ("prepare_local_batch_files", "mc_inputs"),
    ],
)
def test_refresh_requires_new_generation_before_downstream_route(tool, mode):
    s = switched()
    assert "new_model_branch_generation_required" in branch_entry_errors(
        {"tool": tool, "target_ids": ["old_branch"], "parameters": {"mode": mode}}, s
    )
    assert branch_entry_errors({"tool": "generate_branches"}, s) == []
    assert branch_entry_errors({"tool": "pause_search"}, s) == []
    assert (
        branch_entry_errors(
            {"tool": "prepare_local_batch_files", "parameters": {"mode": "model_refresh_inputs"}}, s
        )
        == []
    )


def test_first_search_can_mix_old_and_new_branches_and_choose_dft():
    s = generated()
    assert new_model_search_errors(s, ["b1"]) == []
    assert new_model_search_errors(s, ["old_branch"]) == []
    assert new_model_search_errors(s, ["b1", "old_branch"]) == []
    assert new_model_search_errors(s, dft=True) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_version", "old"),
        ("refresh_plan_checksum", "other"),
        ("new_branch_ids", []),
        ("registered_ids", []),
    ],
)
def test_wrong_or_empty_lineage_cannot_satisfy_generation(field, value):
    s = generated()
    s["generation_history"][-1][field] = value
    assert not new_model_generation_facts(s)["ready"]


def test_baseline_model_and_explicit_current_validation_are_preserved():
    s = {"active_model_version": "base", "generation_history": [{"registered_ids": ["s0"]}]}
    assert new_model_search_errors(s, ["old_branch"]) == []
    s = switched()
    s["training_validation_planning"] = {"status": "active", "scope": {"model_version": "new"}}
    assert new_model_search_errors(s, dft=True) == ["new_model_branch_generation_required"]
    assert new_model_search_errors(s, ["old_branch"]) == ["new_model_branch_generation_required"]


def test_direct_diagnosis_cannot_bypass_prerequisite():
    from phase_agent.tools.relax_mc_batch.plan import diagnose

    with pytest.raises(ValueError, match="new_model_branch_generation_required"):
        diagnose(["old_branch"], switched(), None, {}, "v")


def test_cancelled_bundle_cannot_reactivate_or_rewrite_files(tmp_path):
    from phase_agent.tools.relax_mc_batch.prepare import prepare_batch

    s = switched()
    s["relax_mc_batches"] = {"b": {"status": "cancelled", "plan_checksum": "p"}}
    action = {"parameters": {"relax_mc_preview": {"bundle_id": "b", "checksum": "p"}}}
    before = deepcopy(s)
    r = prepare_batch(
        action=action, context={"event_state": s, "effective_config": {}, "manager": None}
    )
    assert r["status"] == "rejected" and "cancelled" in r["reason"]
    assert s == before and list(tmp_path.iterdir()) == []
    from phase_agent.tools.relax_mc_batch.results import import_results, recover_available

    s["relax_mc_batches"]["b"].update(
        directory=str(tmp_path / "batch"), expected_results_archive=str(tmp_path / "results.zip")
    )
    (tmp_path / "results.zip").write_bytes(b"untrusted late return")
    assert recover_available(s, {}) == s
    with pytest.raises(ValueError, match="cancelled"):
        import_results(action={"parameters": {"bundle_id": "b"}}, context={"event_state": s})


def test_generation_records_only_genuinely_new_branch_ids(monkeypatch):
    from phase_agent.tools.workflows import create_active_learning_handlers as module

    manager = SimpleNamespace(
        data={"branches": {"old_branch": {"P": "O3"}}}, boundary={"P": ["O3"]}
    )

    def generate(*args, **kw):
        manager.data["branches"]["b1"] = {"P": "O3"}
        return {
            "registered": [
                {"branch_id": "old_branch", "structure_id": "s0"},
                {"branch_id": "b1", "structure_id": "s1"},
            ],
            "summary": {"registered_structures": 2, "registered_branches": 2},
            "coverage": {},
        }

    monkeypatch.setattr(module, "run_branch_generation", generate)
    config = {
        "structure_directory": "fake",
        "batch_size": 2,
        "initial_states_per_branch": 1,
        "seed": 0,
    }
    result = module._generate_branches(
        action={
            "tool": "generate_branches",
            "task_key": "g1",
            "parameters": {"quotas": {"coverage": 2}},
        },
        context={
            "event_state": switched(),
            "manager": manager,
            "phase_references": {},
            "effective_config": config,
        },
    )
    record = result["state"]["generation_history"][-1]
    assert record["new_branch_ids"] == ["b1"]
    assert record["model_version"] == "new" and record["refresh_plan_checksum"] == "refresh1"
    assert new_model_generation_facts(result["state"])["ready"]


def test_context_transport_and_cached_snapshot_keep_prerequisite():
    from phase_agent.tools.state.state_manager import agent_state_summary
    from phase_agent.decisions.agent.deepagents_proposal import proposal_material

    s = switched()
    s["current_state_snapshot"] = {"decision_context": {}}
    c = agent_state_summary(s)["decision_context"]
    compact = proposal_material({"decision_context": c})["decision_context"]["new_model_generation"]
    assert compact["post_refresh"] and compact["required"] and not compact["ready"]
    assert compact["generated_branch_ids"] == []


def test_status_reports_the_missing_current_generation_stage():
    from phase_agent.runtime.chat_state_presentation import brief_chat_state

    s = switched()
    s["model_refresh"].update(supplemental_declined_by_user=True, recovered_task_count=285)
    assert "先生成新branch" in brief_chat_state(s)
    s = generated()
    assert "本轮新branch已登记" in brief_chat_state(s)
    s["tasks"] = [{"status": "pending", "stage": "relax_and_feature", "model_version": "new"}]
    assert "等待按需搜索方案" not in brief_chat_state(s)
