"""One real interaction in a fresh process; only external decisions/science are fake."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def main():
    request_path = Path(sys.argv[1])
    request = read_json(request_path, {})
    runtime_path = Path(request["runtime_path"])
    kwargs = read_json(runtime_path, {})
    from phase_agent.configuration.session.save_config_session import save_config_session
    from phase_agent.runtime.runtime_session import load_or_initialize_runtime_session

    session_path = runtime_path.with_name("config_session.json")
    if not session_path.is_file():
        save_config_session(kwargs["config_session"], session_path)
    kwargs["config_session"] = load_or_initialize_runtime_session(
        read_json(kwargs["state_path"], {}),
        state_path=Path(kwargs["state_path"]),
        resolved_session=session_path,
        workspace_defaults={},
    )
    kwargs["config_session_path"] = str(session_path)
    ledger = Path(request["ledger_path"])
    manager = PhaseDataManager.load(ledger)

    def model(payload):
        print("MODEL_REQUEST " + json.dumps(payload, ensure_ascii=False), flush=True)
        response = sys.stdin.readline()
        assert response, "Parent fake model disconnected"
        return json.loads(response)

    kwargs.update(manager=manager, agent_client=model)
    handler = RunWorkflowChatHandler(kwargs)
    if request.get("returned_science"):
        import phase_agent.runtime.main as workflow

        actual_analysis = workflow._analyze_workflow_results
        scientific = request["returned_science"]

        def analyze(frame):
            frame = actual_analysis(frame)
            current = frame["feedback"]["state"]
            current["tasks"][0].update(deepcopy(scientific["task"]))
            current["phase_records"] = deepcopy(scientific["phase_records"])
            current["phase_diagrams"]["mlip"] = deepcopy(scientific["phase_diagram"])
            return frame

        workflow._analyze_workflow_results = analyze
    if request["transport"] == "studio_process":
        from langchain_core.messages import HumanMessage
        from phase_agent.graphs.studio_chat_graph import build_studio_graph
        from phase_agent.runtime import studio_runtime

        config = request_path.parent / "agent_runtime.json"
        write_json(config, {"state_path": str(handler.state_path), "ledger_path": str(ledger)})
        os.environ["PHASE_AGENT_RUNTIME_CONFIG"] = str(config)
        studio_runtime._handler = handler
        graph = build_studio_graph(receipt_path=request_path.parent / "gateway.json")
        result = graph.invoke(
            {"messages": [HumanMessage(content=request["text"], id=request["message_id"])]},
            {"configurable": {"thread_id": request["conversation_id"]}},
        )
        reply = result["messages"][-1].content
    else:
        reply = handler(
            [{"role": "user", "content": request["text"]}],
            conversation_id=request["conversation_id"],
        )
    manager.save(ledger)
    write_json(
        runtime_path,
        {
            key: value
            for key, value in handler.workflow_kwargs.items()
            if key not in {"manager", "agent_client"}
        },
    )
    write_json(request_path.with_suffix(".reply.json"), {"reply": reply, "pid": os.getpid()})
    print("TURN_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
