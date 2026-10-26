"""One JSON containing a DFT selection plan and its prepared task inventory."""

from collections import Counter
from pathlib import Path

from phase_agent.tools.workflows.generation_artifacts import artifact_path, determinant
from phase_agent.tools.relax_mc_batch.files import read, write
from phase_agent.tools.workflows.branch_review_view import raw_record
from phase_agent.tools.workflows.dft_review_view import readable_view
from phase_agent.tools.workflows.compact_review import review_artifacts


def selection_path(action, state):
    path = artifact_path(state, action.get("task_key"), action.get("parameters") or {})
    if path is None:
        return None
    return path.parent.parent / "dft_plans" / path.name.replace("branch_plan_", "dft_selection_")


def proposal_artifacts(proposal, state):
    action = proposal.get("raw_action") or {}
    if (action.get("tool") or proposal.get("recommended_action")) != "select_dft_candidates":
        return {}
    params = action.get("parameters") or proposal.get("action_parameters") or {}
    action = {**action, "parameters": params}
    path = selection_path(action, state)
    if path is None:
        return {}
    from phase_agent.tools.workflows.readonly_views import is_read_only

    if is_read_only():
        return review_artifacts(path, kind="dft", state=state)
    if path.exists():
        saved = read(path)
        original = raw_record(saved)
        if original.get("actual") is not None:
            if original is saved:
                write(path, readable_view(original), sort_keys=False)
            return review_artifacts(path, kind="dft", record=original, state=state)
    # Enrich display metadata from the owned ledger, never create scientific objects.
    root = path.parents[2].resolve()
    config = state.get("confirmed_config") or {}
    storage = config.get("storage") or (config.get("config") or {}).get("storage") or {}
    ledger_path = root / (storage.get("paths") or {}).get(
        "ledger", "workflow_state/ledgers/phase_data.json"
    )
    # Migrated projects retain frozen config paths; use only this state's live binding.
    layout = state.get("workspace_layout") or {}
    runtime_path = root / "agent_runtime.json"
    if layout.get("state_path") and runtime_path.is_file():
        runtime = read(runtime_path)
        if (
            runtime.get("state_path")
            and (root / runtime["state_path"]).resolve() == Path(layout["state_path"]).resolve()
        ):
            if runtime.get("ledger_path"):
                ledger_path = root / runtime["ledger_path"]
    if not ledger_path.resolve().is_relative_to(root):
        raise ValueError("DFT review ledger path is outside project")
    ledger = read(ledger_path) if ledger_path.is_file() else {}
    payload = selection_record(proposal, state, ledger=ledger)
    write(path, readable_view(payload), sort_keys=False)
    return review_artifacts(path, kind="dft", record=payload, state=state)


def selection_record(proposal, state, *, ledger=None):
    """Project saved choices only; no file reads, scientific decisions or mutations."""
    action = proposal.get("raw_action") or {}
    params = action.get("parameters") or proposal.get("action_parameters") or {}
    ledger = ledger or {}
    candidates = {
        r.get("candidate_id") or r.get("structure_id"): r for r in state.get("qbc_candidates") or []
    }
    for row in (proposal.get("decision_context") or {}).get("qbc_candidates") or []:
        if isinstance(row, dict):
            key = row.get("candidate_id") or row.get("structure_id")
            if key in candidates:
                saved = candidates[key]
                sampling = row.get("dft_sampling_evidence") or {}
                if (
                    sampling.get("status") == "available"
                    and (
                        (
                            row.get("structure_sha256")
                            and row.get("structure_sha256") == saved.get("structure_sha256")
                        )
                        or (
                            row.get("structure_path")
                            and row.get("structure_path") == saved.get("structure_path")
                        )
                    )
                    and sampling.get("model_version") == state.get("active_model_version")
                    and sampling.get("hull_version")
                    == (state.get("phase_diagrams", {}).get("mlip") or {}).get("version")
                ):
                    candidates[key] = {**saved, "dft_sampling_evidence": sampling}
    preview = params.get("dft_selection_preview") or params.get("dft_input_preview") or {}
    previews = {r.get("candidate_id"): r for r in preview.get("selected_structures") or []}
    branches = {
        r.get("branch_id"): r
        for r in (proposal.get("decision_context") or {}).get("available_branches") or []
    }
    branches = {**(ledger.get("branches") or {}), **branches}
    task_index = {}
    for task in state.get("tasks") or []:
        if task.get("stage") in {"dft_single_point", "dft_relax"}:
            task_index.setdefault(task.get("structure_id") or task.get("object_id"), []).append(
                {k: task.get(k) for k in ("task_id", "stage", "status", "model_version")}
            )
    rows = []
    for decision in params.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        sid = decision.get("candidate_id")
        candidate = candidates.get(sid) or {}
        evidence = previews.get(sid) or {}
        structure = (ledger.get("structures") or {}).get(sid) or {}
        branch_id = candidate.get("branch_id") or structure.get("branch_id")
        branch = branches.get(branch_id) or {}
        # Context summaries may omit H; the ledger retains the full matrix.
        stored_branch = (ledger.get("branches") or {}).get(branch_id) or {}
        matrix = candidate.get("H") or branch.get("H") or stored_branch.get("H")
        rows.append(
            {
                "candidate_id": sid,
                "branch_id": branch_id,
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "phase": candidate.get("phase") or evidence.get("phase") or branch.get("P"),
                "H": matrix,
                "supercell_det": determinant(matrix),
                "Na_fraction": candidate.get(
                    "x_Na_per_O2", evidence.get("x_Na_per_O2", branch.get("x"))
                ),
                "composition": candidate.get("composition") or branch.get("composition"),
                "atom_count": candidate.get("atom_count"),
                "structure_path": candidate.get("structure_path") or structure.get("source_path"),
                "Ehull": candidate.get("predicted_Ehull", evidence.get("ehull")),
                "Ehull_unit": candidate.get("ehull_unit") or evidence.get("ehull_unit"),
                "phase_diagram_version": candidate.get("phase_diagram_version"),
                "qbc": candidate.get("qbc"),
                "dft_sampling_evidence": candidate.get("dft_sampling_evidence"),
                "training_membership": candidate.get("training_membership", "unknown"),
                "known_DFT_tasks": task_index.get(sid, []),
            }
        )
    counts = Counter(r["action"] for r in rows)
    return {
        "schema_version": 1,
        "status": "awaiting_approval",
        "task_key": action.get("task_key"),
        "model_version": state.get("active_model_version"),
        "config_version": state.get("confirmed_config_version"),
        "purpose": proposal.get("expected_purpose") or action.get("expected_purpose"),
        "reason": action.get("reason"),
        "summary": {
            "selected_count": counts["DFT_SINGLE_POINT"] + counts["DFT_RELAX"],
            "counts_by_action": dict(counts),
        },
        "decisions": rows,
        "input_preview": preview,
        "estimated_cost": proposal.get("estimated_cost"),
        "selection_purposes": (action.get("scientific_decision") or {}).get("selection_purposes"),
        "route_review": (
            action.get("scientific_decision")
            or (proposal.get("decision_summary") or {}).get("scientific_decision")
            or {}
        ),
        "submit_authorized": False,
        "actual": None,
        "note": "null/unknown 表示记录缺失；未见DFT任务不代表从未计算，训练独立性未提供时不作推断。",
    }


def preparation_artifacts(action, state, tasks, validation, prepared=None, manager=None):
    path = selection_path(action, state)
    artifacts = (
        {"json": str(path)}
        if path is not None and path.exists()
        else proposal_artifacts({"raw_action": action}, state)
    )
    if not artifacts:
        return {}
    path = selection_path(action, state)
    payload = raw_record(read(path))
    if manager is not None:
        for row in payload.get("decisions") or []:
            branch = manager.data.get("branches", {}).get(row.get("branch_id")) or {}
            row["H"] = row.get("H") or branch.get("H")
            row["supercell_det"] = determinant(row["H"])
    status = (
        "input_preparation_failed"
        if prepared and prepared.get("status") in {"failed", "rejected"}
        else "prepared"
        if prepared
        else "tasks_created"
    )
    payload.update(
        status=status,
        actual={
            "task_count": len(tasks),
            "tasks": [
                {
                    k: t.get(k)
                    for k in (
                        "task_id",
                        "structure_id",
                        "branch_id",
                        "stage",
                        "structure_path",
                        "planned_relative_cost",
                        "selection_reason",
                        "selection_purpose",
                        "config_version",
                        "model_version",
                    )
                }
                for t in tasks
            ],
            "accepted": validation.get("accepted"),
            "rejected": validation.get("rejected"),
            "input_preparation": {
                k: prepared[k]
                for k in (
                    "status",
                    "reason",
                    "upload_directory",
                    "upload_folder",
                    "directory",
                    "batches",
                )
                if prepared and k in prepared
            },
        },
    )
    write(path, readable_view(payload), sort_keys=False)
    return review_artifacts(path, kind="dft", record=payload, state=state)
