"""Checkpointed training lifecycle with native external-result interrupts."""

from copy import deepcopy
from pathlib import Path
from typing import TypedDict
import hashlib
import json


class TrainingState(TypedDict, total=False):
    business: dict
    config: dict
    job_key: str
    inspection: dict | None
    request: dict
    handoff: dict
    route: str
    node_trace: list[str]


def build_training_handoff_graph(checkpointer=None):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from phase_agent.tools.local.recover_remote_training import inspect_training_results
    from phase_agent.tools.local.training_handoff import (
        prepare_manifest_job,
        validation_request,
        prepare_validation_job,
        recover_validation,
        record_handoff,
    )

    def traced(state, name, **updates):
        return {"node_trace": [*(state.get("node_trace") or [])[-63:], name], **updates}

    def collect(state):
        current = deepcopy(state["business"])
        job = current["remote_finetune_jobs"][state["job_key"]]
        if job.get("activated"):
            return traced(
                state, "collect_training_results", business=current, inspection=None, route="done"
            )
        result = inspect_training_results(job)
        if result is not None:
            version = "remote-" + state["job_key"] + "-" + str(result.get("fingerprint", ""))[:12]
            if current.get("active_model_version") == version:
                job["activated"] = True
                job["training_handoff"] = {
                    "training_fingerprint": result.get("fingerprint"),
                    "stage": "activated",
                    "reason": "模型已激活，转入现有刷新与轮次收益评估流程。",
                }
                return traced(
                    state,
                    "collect_training_results",
                    business=current,
                    inspection=None,
                    route="done",
                )
            job["returned_results"] = result
            if result["status"] != "invalid_results":
                job["status"] = "results_received"
        return traced(
            state,
            "collect_training_results",
            business=current,
            inspection=result,
            request={},
            handoff={},
            route="inspect" if result else "done",
        )

    def check(state):
        result = state["inspection"]
        handoff = {"training_fingerprint": result.get("fingerprint"), "submitted": False}
        status = result["status"]
        if status == "invalid_results":
            handoff.update(
                stage="results_invalid", reason="训练回传待修正：" + "；".join(result["issues"])
            )
        return traced(
            state,
            "check_training_manifest",
            handoff=handoff,
            route={"metadata_required": "manifest", "validation_required": "prerequisites"}.get(
                status, "record"
            ),
        )

    def guarded(state, name, callback):
        try:
            return traced(state, name, **callback())
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
            return traced(
                state,
                name,
                handoff={
                    **state["handoff"],
                    "stage": "handoff_error",
                    "reason": "训练交接待修正：" + str(error),
                },
                route="record",
            )

    def metadata(state):
        return guarded(
            state,
            "prepare_manifest_job",
            lambda: {
                "handoff": prepare_manifest_job(
                    state["business"]["remote_finetune_jobs"][state["job_key"]],
                    state["config"],
                    deepcopy(state["handoff"]),
                ),
                "route": "record",
            },
        )

    def prerequisites(state):
        def run():
            request = validation_request(
                state["business"],
                state["job_key"],
                state["business"]["remote_finetune_jobs"][state["job_key"]],
                state["config"],
                state["inspection"],
            )
            handoff = deepcopy(state["handoff"])
            if request.get("missing"):
                from phase_agent.tools.local.training_cv_review import prepare_cv_review

                review, message = prepare_cv_review(
                    state["business"],
                    state["business"]["remote_finetune_jobs"][state["job_key"]],
                    state["inspection"],
                )
                handoff.update(
                    stage="validation_prerequisites_required",
                    missing=request["missing"],
                    cv_review=review,
                    reason=message,
                )
                current = deepcopy(state["business"])
                settings = state["config"].get("remote_training_validation") or {}
                if settings.get("review_mode", "grouped_cv") == "grouped_cv":
                    from phase_agent.tools.local.training_cv_review import register_cv_candidate

                    register_cv_candidate(
                        current,
                        state["job_key"],
                        current["remote_finetune_jobs"][state["job_key"]],
                        state["inspection"],
                        review,
                        handoff,
                    )
                return {
                    "business": current,
                    "handoff": handoff,
                    "request": request,
                    "route": "record",
                }
            handoff.update(
                request_id=request["plan"]["request_id"],
                candidate_model_version=request["new"]["version"],
            )
            return {
                "handoff": handoff,
                "request": request,
                "route": "recover" if Path(request["returned"]).is_file() else "prepare",
            }

        return guarded(state, "check_validation_prerequisites", run)

    def prepare(state):
        return guarded(
            state,
            "prepare_validation_job",
            lambda: {
                "handoff": prepare_validation_job(
                    state["business"]["remote_finetune_jobs"][state["job_key"]],
                    state["config"],
                    state["request"],
                    deepcopy(state["handoff"]),
                ),
                "route": "record",
            },
        )

    def validate(state):
        def run():
            current, handoff = recover_validation(
                deepcopy(state["business"]), state["request"], deepcopy(state["handoff"])
            )
            return {"business": current, "handoff": handoff, "route": "record"}

        return guarded(state, "validate_and_register_candidate", run)

    def record(state):
        current = record_handoff(deepcopy(state["business"]), state["job_key"], state["handoff"])
        stage = state["handoff"]["stage"]
        route = {
            "awaiting_manifest": "manifest_wait",
            "awaiting_validation": "validation_wait",
            "awaiting_activation_approval": "approval_wait",
        }.get(stage, "input_wait")
        if stage in {"candidate_rejected", "validation_rejected"}:
            route = "done"
        return traced(state, "persist_training_transition", business=current, route=route)

    def waiting(name):
        def node(state):
            answer = interrupt(
                {"kind": name, "job_key": state["job_key"], "handoff": state["handoff"]}
            )
            # Only reconciliation with fresh canonical state can resume. No generic approval value.
            if (
                not isinstance(answer, dict)
                or answer.get("job_key") != state["job_key"]
                or not isinstance(answer.get("business"), dict)
                or not isinstance(answer.get("config"), dict)
            ):
                raise ValueError("training_resume_identity_mismatch")
            return traced(
                state, name, business=answer["business"], config=answer["config"], route="inspect"
            )

        return node

    graph = StateGraph(TrainingState)
    nodes = {
        "collect_training_results": collect,
        "check_training_manifest": check,
        "prepare_manifest_job": metadata,
        "check_validation_prerequisites": prerequisites,
        "prepare_validation_job": prepare,
        "validate_and_register_candidate": validate,
        "persist_training_transition": record,
        "wait_for_manifest_return": waiting("manifest_return"),
        "wait_for_validation_return": waiting("validation_return"),
        "wait_for_activation_decision": waiting("activation_decision"),
        "wait_for_configuration_or_repair": waiting("configuration_or_repair"),
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "collect_training_results")
    graph.add_conditional_edges(
        "collect_training_results",
        lambda s: s["route"],
        {"done": END, "inspect": "check_training_manifest"},
    )
    graph.add_conditional_edges(
        "check_training_manifest",
        lambda s: s["route"],
        {
            "manifest": "prepare_manifest_job",
            "prerequisites": "check_validation_prerequisites",
            "record": "persist_training_transition",
        },
    )
    graph.add_conditional_edges(
        "check_validation_prerequisites",
        lambda s: s["route"],
        {
            "record": "persist_training_transition",
            "prepare": "prepare_validation_job",
            "recover": "validate_and_register_candidate",
        },
    )
    for name in (
        "prepare_manifest_job",
        "prepare_validation_job",
        "validate_and_register_candidate",
    ):
        graph.add_edge(name, "persist_training_transition")
    graph.add_conditional_edges(
        "persist_training_transition",
        lambda s: s["route"],
        {
            "done": END,
            "manifest_wait": "wait_for_manifest_return",
            "validation_wait": "wait_for_validation_return",
            "approval_wait": "wait_for_activation_decision",
            "input_wait": "wait_for_configuration_or_repair",
        },
    )
    for name in (
        "wait_for_manifest_return",
        "wait_for_validation_return",
        "wait_for_activation_decision",
        "wait_for_configuration_or_repair",
    ):
        graph.add_edge(name, "collect_training_results")
    return graph.compile(checkpointer=checkpointer, name="training_lifecycle")


def _run_job(graph, current, config, key, thread):
    from langgraph.types import Command

    invocation = {"configurable": {"thread_id": thread}, "recursion_limit": 32}
    snapshot = graph.get_state(invocation)
    fresh = {
        "business": deepcopy(current),
        "config": deepcopy(config),
        "job_key": key,
        "node_trace": [],
    }
    if snapshot.next and any(task.interrupts for task in snapshot.tasks):
        graph.invoke(Command(resume=fresh), invocation)
    elif snapshot.next:
        # Recover a crash between nodes by rechecking canonical inputs, rather than
        # replaying an unfinished side effect against the checkpoint snapshot.
        graph.update_state(invocation, fresh, as_node="wait_for_configuration_or_repair")
        graph.invoke(None, invocation)
    else:
        graph.invoke(fresh, invocation)
    saved = graph.get_state(invocation)
    return saved.values


def run_training_handoffs(state, config, *, state_path=None, graph=None):
    from phase_agent.tools.local.recover_remote_training import INACTIVE
    from filelock import FileLock
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.memory import InMemorySaver

    current = deepcopy(state)
    waits = []
    keys = [
        key
        for key, job in current.get("remote_finetune_jobs", {}).items()
        if job.get("status") not in INACTIVE
    ]

    def run(saver):
        nonlocal current
        from copy import copy

        execution_graph = copy(graph) if graph is not None else build_training_handoff_graph(saver)
        execution_graph.checkpointer = saver
        for key in keys:
            identity = json.dumps(
                {
                    "state_path": str(Path(state_path).resolve()) if state_path else "ephemeral",
                    "job_key": key,
                    "directory": current["remote_finetune_jobs"][key]["directory"],
                },
                sort_keys=True,
            )
            saved = _run_job(
                execution_graph, current, config, key, hashlib.sha256(identity.encode()).hexdigest()
            )
            current = saved["business"]
            if saved.get("inspection") is not None:
                handoff = current["remote_finetune_jobs"][key].get("training_handoff")
                if handoff and handoff["stage"] not in {
                    "candidate_rejected",
                    "validation_rejected",
                    "activated",
                }:
                    waits.append(handoff)
        return current, waits

    if not state_path:
        return run(InMemorySaver())
    path = Path(state_path).resolve().parent / "langgraph_training.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        with SqliteSaver.from_conn_string(str(path)) as saver:
            return run(saver)
