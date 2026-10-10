from copy import deepcopy
from unittest.mock import patch

from phase_agent.analysis.state.build_decision_context import build_decision_context
from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs
from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
from phase_agent.tools.workflows.run_tool_step import run_tool_step


def test_compaction_keeps_all_candidates_memory_versions_and_request():
    rows = [{"candidate_id": str(i), "predicted_Ehull": .01, "phase": "O3",
             "qbc": {"status": "not_configured"}, "structure": {"sites": [1] * 100},
             "phase_diagram_version": "h2"} for i in range(240)]
    memory = {"human_system_knowledge": ["预算3000"], "physical_priors": ["层状"],
              "frozen_parameter_advice": ["同TM"], "search_rules": ["考虑改善幅度"]}
    context = {"qbc_candidates": rows, "long_term_memory": memory,
               "long_term_human_advice": {"items": ["预算3000"], "version": 3},
               "recent_experience": {"rewards": [{"ehull_improvement": .02, "batch_id": "b"}]}}
    payload = {"state": {"qbc_candidates": rows, "decision_context": context,
                         "long_term_memory": memory, "user_message": "探索P3"},
               "decision_context": context}
    original = deepcopy(payload)
    result = prepare_llm_request(payload)
    assert payload == original
    assert "qbc_candidates" not in result["state"]
    assert len(result["decision_context"]["qbc_candidates"]) == 240
    assert "structure" not in result["decision_context"]["qbc_candidates"][0]
    assert result["decision_context"]["long_term_memory"] == memory
    assert result["decision_context"]["recent_experience"] == context["recent_experience"]
    assert result["state"]["user_message"] == "探索P3"
    assert result["transport_metrics"]["compacted_chars"] < result["transport_metrics"]["original_chars"]


def test_old_scientific_experience_survives_status_messages():
    records = [{"record_id": "mc", "final_action": {"tool": "allocate_mc_bohb",
                 "reason": "探索", "evidence_refs": ["batch:b"]}}]
    records += [{"record_id": str(i), "final_action": {"tool": "status"}} for i in range(20)]
    context = build_decision_context({"action_records": records,
        "qbc_candidates": [{"candidate_id": str(i)} for i in range(240)]})
    assert len(context["qbc_candidates"]) == 240
    assert any(row["record_id"] == "mc" for row in context["recent_experience"]["actions"])


def test_soft_coverage_warns_without_replacing_llm_selection():
    state = {"qbc_candidates": [{"candidate_id": "a", "phase": "O3", "atom_count": 48},
                                 {"candidate_id": "b", "phase": "P3", "atom_count": 48}]}
    config = {"qbc": default_dft_decision_config()}
    action = {"parameters": {"decisions": [{"candidate_id": "a", "action": "DFT_SINGLE_POINT",
                                             "reason": "历史改善优先"}]}}
    preview, error = preview_dft_inputs(action, state, config)
    assert error is None
    assert preview["candidate_ids"] == ["a"]
    assert "P3" in preview["warnings"][0]


def test_configured_trigger_rejects_instead_of_changing_action():
    config = default_dft_decision_config()
    config["extreme_uncertainty"] = {"f_std_max": .1, "hard_action": "DFT_SINGLE_POINT"}
    decision = {"candidate_id": "a", "action": "DEFER", "reason": "暂缓"}
    result = validate_dft_agent_decisions({"decisions": [decision]},
        [{"candidate_id": "a", "f_std_max": 1}], {}, config=config,
        config_version="v", remaining_budget=3000)
    assert result["accepted"] == []
    assert result["rejected"][0]["action"] == "DEFER"
    assert decision["action"] == "DEFER"


def test_failed_validation_returns_to_llm_and_requires_approval():
    config = {"agent": {"allowed_tools": ["select_dft_candidates"]}}
    action = {"tool": "select_dft_candidates", "budget": 1,
              "parameters": {"decisions": [{"candidate_id": "a", "action": "DFT_RELAX"}]}}
    revised = deepcopy(action)
    revised["parameters"]["decisions"][0]["action"] = "DFT_SINGLE_POINT"
    preview = {"relative_cost": 30, "task_count": 1, "workload": []}
    with patch("phase_agent.decisions.agent.choose_debug_next_action.choose_debug_next_action", return_value=None),\
         patch("phase_agent.tools.workflows.run_tool_step.propose_agent_tool_action", return_value=action),\
         patch("phase_agent.tools.workflows.attach_dft_preview.preview_dft_inputs", side_effect=[(None, "dft_budget"), (preview, None)]),\
         patch("phase_agent.tools.workflows.run_tool_step.revise_tool_proposal", return_value={"revision_status": "revised", "action": revised}) as revise,\
         patch("phase_agent.tools.workflows.dft_template_review.gate_template", side_effect=lambda proposal, *args, **kw: (proposal, False)):
        result = run_tool_step({}, {"config": config},
            registry={"select_dft_candidates": {"handler": lambda **kw: None}},
            agent_client=lambda p: {}, execution_mode="interactive")
    assert revise.call_count == 1
    assert "dft_validation_feedback" in revise.call_args.kwargs["state"]["decision_context"]
    assert result["status"] == "awaiting_approval"
    assert not result.get("execution_result")
