"""Exercise chat _run, stale durable lifecycle, actual training analysis and saved approval."""

from pathlib import Path
import pytest

from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.graphs.project.graph import build_react_lifecycle_graph
from phase_agent.graphs.project.runner import run_search_workflow_graph
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate
from tests.test_training_return_contract import delivered, training_analysis_case


@pytest.mark.parametrize("message", ["开始", "继续", "已回收"])
def test_all_chat_advances_reconcile_delivery_before_stale_checkpoint(tmp_path, message):
    state = training_analysis_case(tmp_path)
    delivered_state, action, execution = delivered()
    job = state["remote_finetune_jobs"]["j"]
    transfer = {
        "folder": str(tmp_path / "input"),
        "import_directory": job["directory"],
        "package_fingerprint": "bound",
    }
    job["handoff"] = transfer
    execution["result"]["handoffs"] = [transfer]
    state.update(
        {
            k: delivered_state[k]
            for k in [
                "confirmed_config_version",
                "budget_reservations",
                "effective_decisions",
                "execution_recovery_report",
            ]
        }
    )
    state["action_records"] = [
        {"record_id": "old-preparation", "final_action": action, "execution_result": execution}
    ]
    state_path = tmp_path / "state.json"
    checkpoint = tmp_path / "lifecycle.sqlite"
    write_json(state_path, state)
    executed = []

    def initialize():
        current = read_json(state_path, {})
        return {
            "_workflow_prepared": True,
            "pre_reconciled": {"state": current},
            "effective_config": {},
            "state_path": str(state_path),
            "snapshot": {},
            "execution_mode": "interactive",
            "recovery_question": None,
            "recovered_count": 0,
            "collection_report": {},
            "manual_wait": None,
            "rebuilding": False,
            "runtime_adapters": {"user_message": message},
        }

    def analyze(frame):
        frame["feedback"] = {"state": frame["pre_reconciled"]["state"]}
        return frame

    def forbidden(frame):
        executed.append(True)
        raise AssertionError("Must stop at the saved model-activation approval")

    graph = build_react_lifecycle_graph()
    # Preserve a real unsafe old graph frame; the new request must never replay its finalize.
    with CompactSqliteSaver.from_conn_string(str(checkpoint)) as saver:
        from copy import copy

        seeded = copy(graph)
        seeded.checkpointer = saver
        seeded.update_state(
            {"configurable": {"thread_id": "project"}},
            {"request_id": "old-request", "business_frame": {}},
            as_node="react",
        )
        assert seeded.get_state({"configurable": {"thread_id": "project"}}).next == ("finalize",)

    def workflow(**kwargs):
        return run_search_workflow_graph(
            workflow_graph=graph,
            checkpoint_path=checkpoint,
            business_state_path=state_path,
            invocation_id=kwargs["invocation_id"],
            project_thread="project",
            human_feedback=kwargs.get("human_feedback"),
            initialize=initialize,
            collect=lambda f: f,
            analyze=analyze,
            wait=_workflow_wait_gate,
            assess=forbidden,
            act=forbidden,
            finalize=forbidden,
        )

    handler = RunWorkflowChatHandler({"state_path": str(state_path)}, workflow=workflow)
    result = handler._run("new-request", None, message)
    assert result["status"] == "training_handoff"
    assert result["training_handoffs"][0]["stage"] == "awaiting_activation_approval"
    assert result["training_handoffs"][0]["next_epoch"] == "epoch2"
    assert Path(result["training_handoffs"][0]["training_report_path"]).is_file()
    current = read_json(state_path, {})
    assert current["budget_reservations"]["prepare"]["status"] == "settled"
    assert current["active_model_version"] == "base"
    assert not executed
    # Durable duplicate delivery returns its own prior result without redoing scientific actions.
    repeated = handler._run("new-request", None, message)
    assert repeated["status"] == "training_handoff" and not executed
