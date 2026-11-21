"""Select a scientific diagram bound to one active MLIP model."""


def active_model_diagram(state, fallback=None):
    model = state.get("active_model_version")
    diagrams = state.get("phase_diagrams") or state.get("phase_diagram_state") or {}
    diagrams = diagrams.get("diagrams", diagrams)
    global_diagram = fallback if fallback is not None else (diagrams.get("mlip") or {})
    independent = (state.get("phase_diagrams_by_model") or {}).get(model) if model else None
    if independent:
        if independent.get("model_version") != model:
            raise ValueError("当前模型的独立相图身份不匹配")
        if any(
            row.get("source_version") not in {None, model}
            for row in independent.get("entries") or []
        ):
            raise ValueError("当前模型的独立相图含其他模型数据")
        return independent
    refresh = state.get("model_refresh") or {}
    if (
        model
        and (
            global_diagram.get("model_version") not in {None, model}
            or any(
                row.get("source_version") not in {None, model}
                for row in global_diagram.get("entries") or []
            )
        )
        and refresh.get("new_model_version") == model
        and refresh.get("status") in {"approval_required", "pending", "running", "waiting_results"}
    ):
        # The retained previous hull is a baseline, not a current-model result.
        return {
            "status": "pending_refresh",
            "model_version": model,
            "reason": "current_model_refresh_not_completed",
        }
    sources = {
        row.get("source_version")
        for row in global_diagram.get("entries") or []
        if row.get("source_version")
    }
    if global_diagram.get("mixed_model_versions") or (
        model and any(source != model for source in sources)
    ):
        raise ValueError("总相图混有其他模型，缺少当前模型独立相图")
    return global_diagram


def active_model_summary(state):
    """Expose a verified, compact model-specific hull identity to both entry points."""
    model = state.get("active_model_version")
    independent = (state.get("phase_diagrams_by_model") or {}).get(model) if model else None
    if not independent:
        return None
    diagram = active_model_diagram(state)
    if diagram.get("status") != "completed" or not diagram.get("version"):
        return None
    return {key: diagram[key] for key in ("status", "version", "energy_basis_id") if key in diagram}


def decision_diagram(state, fallback=None):
    """Prefer the completed latest-per-structure hull for scientific decisions."""
    model = state.get("active_model_version")
    diagrams = state.get("phase_diagrams") or state.get("phase_diagram_state") or {}
    diagrams = diagrams.get("diagrams", diagrams)
    latest = fallback if fallback is not None else (diagrams.get("mlip") or {})
    refresh = state.get("model_refresh") or {}
    waiting = refresh.get("new_model_version") == model and refresh.get("status") in {
        "approval_required",
        "pending",
        "running",
        "waiting_results",
    }
    if (
        model
        and not waiting
        and latest.get("model_version") == model
        and latest.get("status") == "completed"
        and latest.get("energy_policy") == "latest_valid_per_structure_v1"
    ):
        return latest
    return active_model_diagram(state, fallback)


def selection_diagram(state):
    """Use the frozen baseline only for an explicitly bound recovery cycle."""
    raw = (state.get("phase_diagrams") or {}).get("mlip") or {}
    cycle = state.get("extrapolation_recovery_cycle") or {}
    if (
        cycle.get("status") == "awaiting_reviewed_dft_labels"
        and state.get("active_model_version") == cycle.get("invalid_model_version")
        and raw.get("model_version") == cycle.get("baseline_model_version")
        and raw.get("version") == cycle.get("baseline_hull_version")
    ):
        return raw
    return decision_diagram(state)


def selection_state(state):
    """Project the selected hull without changing model-specific evidence."""
    return {
        **state,
        "phase_diagrams": {
            **(state.get("phase_diagrams") or {}),
            "mlip": selection_diagram(state),
        },
    }


def active_model_state(state):
    """Project a current-model diagram without mutating the persisted state."""
    return {
        **state,
        "phase_diagrams": {
            **(state.get("phase_diagrams") or {}),
            "mlip": active_model_diagram(state),
        },
    }
