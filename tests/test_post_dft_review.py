from decision_layer.agent.post_dft_review import valid_post_dft_review
from decision_layer.agent.propose_tool_action import propose_agent_tool_action


def review():
    return {"choice": "search", "stop_status": "continue", "finetune_recommendation": "defer", "error_assessment": "能量与力误差需结合用途判断",
            "dft_assessment": "已有结构池不能补齐合成覆盖缺口，暂不补DFT",
            "convergence_assessment": "覆盖缺口存在，尚不能证明收敛",
            "coverage_assessment": "合成证据表明组成覆盖不足",
            "finetune_assessment": "暂缓微调以补充代表性数据，不以未启用为理由",
            "search_assessment": "优先补充缺失组成，仍需批准",
            "reference_assessment": "结合合成覆盖记录；未设置硬性误差阈值",
            "round_findings": "合成记录不支持量化稳定相收益，先说明证据缺口",
            "limitations": "部分组成缺少代表性数据"}


def parameters():
    return {"quotas": {"coverage": 10}, "total_quota": 10,
            "generation_plan": [{"strategy": "coverage", "quota": 10, "phase": "all", "reason": "补覆盖"}]}


def test_review_requires_both_alternatives_and_matching_choice():
    action = {"tool": "generate_branches", "post_dft_review": review(), "parameters": parameters()}
    assert valid_post_dft_review(action)
    action["post_dft_review"].pop("finetune_assessment")
    assert not valid_post_dft_review(action)
    assert not valid_post_dft_review({"tool": "update_mlip", "post_dft_review": review()})


def test_input_only_finetune_accepts_legacy_wire_label_not_search():
    from decision_layer.agent.post_dft_review import request_review_repair
    fields = review()
    fields.update(choice="revise_strategy", finetune_recommendation="now")
    action = {"tool": "update_mlip", "parameters": {"prepare_inputs_only": True}, "post_dft_review": fields}
    def forbidden(payload):
        raise AssertionError("equivalent label should not need another LLM call")
    normalized = request_review_repair(forbidden, {}, action)
    assert normalized["post_dft_review"]["choice"] == "finetune"
    assert action["post_dft_review"]["choice"] == "revise_strategy"
    fields["choice"] = "search"
    from decision_layer.agent.post_dft_review import normalize_review_choice
    assert not valid_post_dft_review(normalize_review_choice(action))


def test_repair_response_tool_alias_is_normalized():
    from decision_layer.agent.post_dft_review import request_review_repair
    fields = review()
    fields.update(choice="update_mlip", finetune_recommendation="now")
    repaired = request_review_repair(lambda payload: {"tool": "update_mlip",
        "parameters": {"prepare_inputs_only": True}, "post_dft_review": fields},
        {"instruction": "test", "allowed_tools": ["update_mlip"]}, {"tool": "update_mlip"})
    assert repaired["post_dft_review"]["choice"] == "finetune"


def test_generic_branch_reason_cannot_pass_post_dft_gate():
    state = {"available_branches": [{"branch_id": "test"}],
             "decision_context": {"post_dft_assessment": {"status": "evaluated"}}}
    action = propose_agent_tool_action(state, allowed_tools=["generate_branches", "pause_search"],
        agent_client=lambda request: {"tool": "generate_branches", "reason": "扩展覆盖"})
    assert action["tool"] == "pause_search"
    assert "DFT 后方案修正失败" in action["fallback_reason"]


def test_review_reports_specific_missing_field_and_choice_conflict():
    from decision_layer.agent.post_dft_review import post_dft_review_errors
    fields = review()
    fields.pop("limitations")
    fields["finetune_recommendation"] = "now"
    errors = post_dft_review_errors({"tool": "generate_branches", "parameters": parameters(),
                                     "post_dft_review": fields})
    assert any("limitations" in e for e in errors)
    assert any("搜索需明确defer" in e for e in errors)


def test_llm_repairs_once_with_exact_feedback_and_consistent_schema():
    requests = []
    def client(payload):
        requests.append(payload)
        if len(requests) == 1:
            return {"tool": "generate_branches", "parameters": parameters()}
        assert "post_dft_review: 缺少分析对象" in payload["validation_errors"]
        assert "with only tool" not in payload["instruction"]
        return {"tool": "generate_branches", "parameters": parameters(), "post_dft_review": review()}
    result = propose_agent_tool_action({"available_branches": [{}],
        "decision_context": {"post_dft_assessment": {"status": "evaluated"}}},
        allowed_tools=["generate_branches", "adjust_strategy", "pause_search"], agent_client=client)
    assert result["tool"] == "generate_branches"
    assert len(requests) == 2


def test_repair_failure_is_bounded_and_not_approved():
    requests = []
    def client(payload):
        requests.append(payload)
        return {"tool": "generate_branches", "parameters": parameters()}
    result = propose_agent_tool_action({"available_branches": [{}],
        "decision_context": {"post_dft_assessment": {"status": "evaluated"}}},
        allowed_tools=["generate_branches", "pause_search"], agent_client=client)
    assert len(requests) == 2
    assert result["tool"] == "pause_search"
    assert "post_dft_review" in result["fallback_reason"]


def test_repair_usage_includes_both_calls():
    from decision_layer.agent.post_dft_review import request_validated_action
    calls = []
    def client(payload):
        calls.append(payload)
        return {"tool": "generate_branches", "parameters": parameters(),
                "post_dft_review": review() if len(calls) == 2 else None,
                "_llm_usage": {"calls": 1, "input_tokens": 10, "output_tokens": 5, "cost": None}}
    result = request_validated_action(client, {"instruction": "Return JSON", "allowed_tools": ["generate_branches"],
        "decision_context": {"post_dft_assessment": {"status": "evaluated"}}})
    assert result["_llm_usage"]["calls"] == 2
    assert result["_llm_usage"]["input_tokens"] == 20


def test_direct_finetune_recommendation_and_conditional_enablement_notice():
    from run.post_dft_presentation import post_dft_review_lines
    fields = review()
    text = post_dft_review_lines({"post_dft_review": fields}, finetune_enabled=False)
    assert "建议暂缓微调。原因：" in text
    assert "批准启用" not in text
    fields["finetune_recommendation"] = "now"
    text = post_dft_review_lines({"post_dft_review": fields}, finetune_enabled=False)
    assert "建议微调。原因：" in text and "批准后仅生成超算训练提交文件" in text
    assert "批准启用微调" not in text
    assert "批准启用" not in post_dft_review_lines({"post_dft_review": fields}, finetune_enabled=True)
    fields["finetune_recommendation"] = "insufficient_evidence"
    assert "暂不能判断" in post_dft_review_lines({"post_dft_review": fields})


def test_review_is_presented_not_only_purpose():
    from run.workflow_reply_presentation import format_workflow_reply
    proposal = {"recommended_action": "generate_branches", "expected_purpose": "扩展覆盖",
                "raw_action": {"tool": "generate_branches", "post_dft_review": review(), "parameters": parameters()},
                "action_parameters": parameters(), "estimated_cost": {}, "calculation_plan": {}}
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal}, "state.json")
    for label in ("本轮发现", "微调取舍", "新branch取舍", "补DFT取舍", "收敛与停止", "数据局限"):
        assert label in text
    assert text.index("本轮总结") < text.index("下一步建议") < text.index("回复“同意”")
    assert text.count("微调取舍") == 1


def test_tradeoff_without_round_summary_is_not_complete():
    fields = review()
    fields.pop("round_findings")
    assert not valid_post_dft_review({"tool": "generate_branches", "post_dft_review": fields})


def test_analysis_report_identifies_saved_phase_diagram_without_inference():
    from run.post_dft_presentation import post_dft_lines
    from tests.test_dft_comparison_csv import add_result
    state = {}
    add_result(state)
    assert "DFT 相图：尚无已保存 CSV" in post_dft_lines(state)
    state["phase_diagrams"] = {"dft": {"csv_path": "synthetic/dft.csv"}}
    assert "DFT 相图：`synthetic/dft.csv`" in post_dft_lines(state)
