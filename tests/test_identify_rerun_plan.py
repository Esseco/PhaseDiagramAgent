import json

from phase_agent.tools.local.identify_rerun_plan import identify_rerun_plan
from phase_agent.runtime.agent_api import RunWorkflowChatHandler


def _record(record_id, tool, *, mode=None, status="completed"):
    parameters = {"mode": mode} if mode else {}
    return {"record_id": record_id, "status": status,
            "final_action": {"tool": tool, "task_key": f"key-{record_id}",
                             "parameters": parameters}}


def _state():
    return {"action_records": [
                _record("gen-1", "generate_branches"),
                _record("relax-1", "prepare_local_batch_files", mode="relax_inputs", status="prepared"),
                _record("mc-1", "allocate_mc_bohb"),
            ],
            "branch_batch": {"branch_ids": ["B-1"]},
            "tasks": [
                {"task_id": "R-1", "branch_id": "B-1", "stage": "relax_and_feature",
                 "input_path": "missing-relax.json"},
                {"task_id": "M-1", "branch_id": "B-1", "stage": "deep_search",
                 "input_path": "missing-mc.json", "result_path": "missing-result.json"},
            ]}


def test_specific_generation_action_is_identified_without_execution():
    plan = identify_rerun_plan("重新生成当前轮Relax输入", _state())
    assert plan["status"] == "identified"
    assert plan["round_anchor_record_id"] == "gen-1"
    assert plan["action"]["record_id"] == "relax-1"
    assert plan["affected_by_stage"] == {"relax_and_feature": 1}


def test_ambiguous_generation_lists_action_candidates():
    plan = identify_rerun_plan("重新生成当前轮任务", _state())
    assert plan["status"] == "ambiguous"
    assert [row["record_id"] for row in plan["candidates"]] == ["gen-1", "relax-1", "mc-1"]


def test_read_result_is_workflow_operation_not_fabricated_action():
    plan = identify_rerun_plan("重新读取当前轮MC结果", _state())
    assert plan["status"] == "identified"
    assert plan["read_operation"] == "collect_results_with_report"
    assert plan["task_count"] == 1
    assert plan["result_file_count"] == 0


def test_explicit_round_requires_recorded_anchor():
    plan = identify_rerun_plan("重新生成第2轮branch", _state())
    assert plan["status"] == "round_unknown"


def test_multiple_rounds_require_a_round_choice():
    state = _state()
    state["action_records"] += [_record("gen-2", "generate_branches")]
    plan = identify_rerun_plan("重新生成branch", state)
    assert plan["status"] == "round_ambiguous"
    assert [row["round_number"] for row in plan["candidate_rounds"]] == [1, 2]
    previous = identify_rerun_plan("重新生成上一轮branch", state)
    assert previous["round_anchor_record_id"] == "gen-1"


def test_read_state_identifies_read_operation_only():
    plan = identify_rerun_plan("重新读取当前轮状态", _state())
    assert plan["status"] == "identified"
    assert plan["read_operation"] == "read_state"


def test_chat_only_identifies_and_does_not_call_workflow(tmp_path):
    path = tmp_path / "state.json"
    state = _state()
    path.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    handler = RunWorkflowChatHandler({"state_path": str(path)},
        workflow=lambda **kwargs: calls.append(kwargs))
    reply = handler([{"role": "user", "content": "重新生成当前轮Relax输入"}])
    assert "relax-1" in reply
    assert "尚未删除文件" in reply
    assert calls == []
    assert json.loads(path.read_text(encoding="utf-8")) == state
