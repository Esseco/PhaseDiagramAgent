"""Keep obsolete parallel controllers out of the production architecture."""

from pathlib import Path
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.graphs.dialogue.graph import build_project_turn_graph


def test_handler_has_no_legacy_base_and_graph_owns_all_turns():
    assert RunWorkflowChatHandler.__bases__ == (object,)
    graph = build_project_turn_graph()
    assert {"prepare_context", "decide", "review", "scientific_workflow"} <= set(graph.nodes)
    context = dict(graph.get_subgraphs())["prepare_context"]
    assert {"restore_history", "finetune", "mc", "dft_recovery"} <= set(context.nodes)
    source = Path(__file__).resolve().parents[1]
    for name in (
        "phase_agent/runtime/legacy_chat_application.py",
        "phase_agent/graphs/search_workflow_graph.py",
        "phase_agent/runtime/chat_intent_routing.py",
        "phase_agent/decisions/agent/resolve_chat_intent.py",
        "phase_agent/decisions/agent/classify_config_edit_intent.py",
        "phase_agent/runtime/run_pipeline.py",
        "phase_agent/runtime/search_iteration.py",
        "phase_agent/runtime/run_active_learning_cycle.py",
        "phase_agent/runtime/resume_pipeline.py",
        "phase_agent/runtime/run_confirmed_active_learning_cycle.py",
        "phase_agent/runtime/run_confirmed_pipeline.py",
    ):
        assert not (source / name).exists()


def test_runtime_composition_has_no_second_intent_client():
    from phase_agent.runtime import runtime_client_settings

    assert not hasattr(runtime_client_settings, "intent_client_settings")


def test_dft_regeneration_needs_fresh_explicit_confirmation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from phase_agent.graphs.dialogue.maintenance.rerun import handle, CONFIRM
    from phase_agent.tools.local import identify_rerun_plan, regenerate_dft_files

    plan = {
        "status": "identified",
        "intent": "generate",
        "action": {"tool": "select_dft_candidates"},
    }
    monkeypatch.setattr(identify_rerun_plan, "identify_rerun_plan", lambda *args: dict(plan))
    monkeypatch.setattr(identify_rerun_plan, "is_rerun_plan_request", lambda *args: True)
    monkeypatch.setattr(identify_rerun_plan, "format_rerun_plan", lambda p: "重生成两个输入")
    execute = Mock(return_value={"task_count": 2, "backup_directory": "backup"})
    monkeypatch.setattr(regenerate_dft_files, "regenerate_dft_files", execute)
    handler = SimpleNamespace(
        state_path=tmp_path / "state.json",
        workflow_kwargs={"run_config": {"upload_batches_directory": str(tmp_path)}},
    )
    state = {}
    assert "尚未修改文件" in handle(handler, "重新生成当前轮DFT", state, [], None)
    execute.assert_not_called()
    plan["affected_task_count"] = 3
    assert "状态已变化" in handle(handler, CONFIRM, state, [], None)
    execute.assert_not_called()
    handle(handler, "重新生成当前轮DFT", state, [], None)
    assert "未提交作业" in handle(handler, CONFIRM, state, [], None)
    assert execute.call_count == 1
    handle(handler, CONFIRM, state, [], None)
    assert execute.call_count == 1
