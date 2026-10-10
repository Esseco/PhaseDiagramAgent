"""Studio status graph: projects saved evidence without scientific execution."""

import os
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ProjectStatus(TypedDict, total=False):
    project_name: str
    project_directory: str
    scientific_progress: dict
    current_stage: str
    agent_recommendations: list[dict]
    awaiting_approval: dict
    process_url: str
    conversation_management_url: str


def read_project_status(state):
    from phase_agent.runtime.scientific_progress import scientific_progress

    config = os.environ.get("PHASE_AGENT_RUNTIME_CONFIG")
    if not config:
        return {
            "scientific_progress": {},
            "current_stage": "项目配置未指定",
            "agent_recommendations": [],
            "awaiting_approval": {},
            "process_url": "",
        }
    project = Path(config).resolve().parent
    progress = scientific_progress(config)
    directions = progress.get("directions") or []
    return {
        "project_name": project.name,
        "project_directory": str(project),
        "scientific_progress": progress,
        "current_stage": progress.get("summary", "项目状态尚未登记"),
        "agent_recommendations": [row["proposal"] for row in directions if not row["activated"]],
        "awaiting_approval": {
            "reviews": progress.get("review_requests") or [],
            "directions": [
                row["proposal"]
                for row in directions
                if row.get("status") == "awaiting_approval"
                and row.get("waiting_at") != "execution_plan_ready"
                and not row["activated"]
            ],
            "execution": progress.get("execution_approvals") or {},
        },
        "process_url": "/phase/process",
        "conversation_management_url": "http://127.0.0.1:"
        + os.environ.get("PHASE_STUDIO_PORT", "2024")
        + "/phase/conversations",
    }


def build_status_graph():
    graph = StateGraph(ProjectStatus)
    graph.add_node("read_persisted_project_status", read_project_status)
    graph.add_edge(START, "read_persisted_project_status")
    graph.add_edge("read_persisted_project_status", END)
    return graph.compile(name="project_status")


graph = build_status_graph()
