"""Publish inspectable memory views; state remains the single authoritative source."""

from pathlib import Path


def publish_memory_views(state, state_path):
    source = Path(state_path).resolve()
    if source.name != "state.json" or source.parent.name not in {
        "runtime",
        "current",
        "workflow_state",
    }:
        return
    from phase_agent.tools.step_runner.file_protocol import write_json

    root = source.parent.parent / (
        "agent_memory" if source.parent.name == "workflow_state" else "memory"
    )
    metadata = {
        "source_state": str(source),
        "generated_view": True,
        "editing": "通过Agent的记忆审阅/批准流程修改；本视图不反向导入，避免两个事实源。",
    }
    views = {
        "decision_memory.json": state.get("decision_memory") or {},
        "memory_candidates.json": state.get("memory_candidates") or [],
        "skill_drafts.json": {
            key: value
            for key, value in state.items()
            if "skill" in key and key not in {"skills_directory"}
        },
    }
    for name, payload in views.items():
        write_json(root / name, {**metadata, "data": payload})
