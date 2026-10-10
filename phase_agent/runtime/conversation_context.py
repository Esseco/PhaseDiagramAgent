"""Small allowlisted facts for conversation, never raw configs or structures."""

from collections import Counter
import json


def conversation_context(state, messages=()):
    model = state.get("active_model") or {}
    jobs = state.get("remote_finetune_jobs") or {}
    pending = state.get("pending_execution_policies") or {}
    artifacts = []
    for job in list(jobs.values())[-2:]:
        artifacts.append(
            {
                key: job.get(key)
                for key in (
                    "directory",
                    "status",
                    "original_model_version",
                    "submitted",
                    "activated",
                )
            }
        )
    diagrams = {
        method: {key: value.get(key) for key in ("csv_path", "version", "status")}
        for method, value in (state.get("phase_diagrams") or {}).items()
        if isinstance(value, dict)
    }
    proposals = []
    for row in list(pending.values())[:2]:
        proposal = row.get("agent_proposal") or {}
        action = proposal.get("raw_action") or {}
        proposals.append(
            {
                "tool": proposal.get("recommended_action") or action.get("tool"),
                "purpose": str(proposal.get("purpose") or "")[:400],
            }
        )
    history = []
    for message in list(messages)[-4:]:
        if isinstance(message, dict) and message.get("role") in {"user", "assistant"}:
            content = message.get("content")
            if isinstance(content, str):
                history.append({"role": message["role"], "content": content[:450]})
    context = {
        "model_version": model.get("version") or state.get("active_model_version"),
        "task_counts": dict(
            Counter(row.get("status", "unknown") for row in state.get("tasks") or [])
        ),
        "training_artifacts": artifacts,
        "phase_diagrams": diagrams,
        "pending_proposals": proposals,
        "training_inputs_changed": bool(state.get("finetune_input_conflict")),
        "recent_conversation": history,
        "training_waits": [
            {
                "stage": (job.get("training_handoff") or {}).get("stage"),
                "missing": (job.get("training_handoff") or {}).get("missing", [])[:4],
            }
            for job in list(jobs.values())[-2:]
        ],
        "pending_count": len(pending),
        "candidate_reviews": [
            {
                "version": str(version)[:180],
                "status": row.get("status"),
                "passed": (row.get("validation") or {}).get("passed"),
            }
            for version, row in list((state.get("candidate_models") or {}).items())[-3:]
        ],
    }
    # Fixed transport budget, without a summarization LLM call. Current wait facts
    # survive before historical messages and paths; full evidence remains on disk.
    while len(json.dumps(context, ensure_ascii=False)) > 3600 and context["recent_conversation"]:
        context["recent_conversation"].pop(0)
    if len(json.dumps(context, ensure_ascii=False)) > 3600:
        context["phase_diagrams"] = {}
        context["training_artifacts"] = []
    if len(json.dumps(context, ensure_ascii=False)) > 3600:
        context["training_waits"] = [{"stage": row["stage"]} for row in context["training_waits"]]
        context["pending_proposals"] = [
            {"tool": row["tool"]} for row in context["pending_proposals"]
        ]
    if len(json.dumps(context, ensure_ascii=False)) > 3600:
        context = {
            "model_version": str(context["model_version"])[:180],
            "pending_count": len(pending),
            "training_waits": [
                {"stage": str(row["stage"])[:80]} for row in context["training_waits"]
            ],
            "candidate_reviews": context["candidate_reviews"],
            "recent_conversation": history[-2:],
        }
    return context
