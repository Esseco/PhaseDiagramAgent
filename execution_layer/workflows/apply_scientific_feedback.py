"""Write recovered calculation results to the ledger and refresh feedback."""

from __future__ import annotations

from copy import deepcopy

from analysis_layer.feedback.calculate_search_reward import calculate_search_reward
from analysis_layer.state.summarize_agent_state import summarize_agent_state
from analysis_layer.phase.update_phase_diagram import update_phase_diagram
from analysis_layer.phase.refresh_identified_phases import refresh_identified_phases
from analysis_layer.phase.ensure_phase_identification import ensure_phase_identification
from data_layer.ledger.collect_calculation_results import collect_calculation_results
from data_layer.ledger.coverage_report import coverage


TERMINAL = {"completed", "failed", "timeout", "cancelled"}


def apply_scientific_feedback(
    state,
    recovered_results,
    *,
    manager,
    ledger_path=None,
    phase_diagram_directory=None,
    final_frame_mlip_evaluator=None,
    active_model_version=None,
    phase_references=None,
    phase_identification_cache_path=None,
):
    """Persist each terminal result once, then rebuild hull/reward summaries."""
    current = deepcopy(state)
    processed = current.setdefault("feedback_processed_task_ids", [])
    current.setdefault("phase_records", [])
    current.setdefault("phase_diagrams", {})
    current.setdefault("phase_diagrams_by_model", {})
    current.setdefault("reward_states", {})
    current.setdefault("rewards", [])
    accepted, rejected, feedback_rows = [], [], []
    if manager is None:
        rejected = [
            {"task_id": row.get("task_id"), "reason": "manager_unavailable"}
            for row in (recovered_results or []) if row.get("status") in TERMINAL
        ]
        current["agent_state_summary"] = summarize_agent_state(current)
        return {"state": current, "recorded": [], "rejected": rejected, "feedback_count": 0}
    for incoming in recovered_results or []:
        result = deepcopy(incoming)
        task_id = result.get("task_id")
        if not task_id or task_id in processed or result.get("status") not in TERMINAL:
            continue
        structure_id = result.get("structure_id") or result.get("object_id")
        if structure_id not in manager.data.get("structures", {}):
            rejected.append({"task_id": task_id, "reason": "unknown_structure"})
            continue
        if result.get("stage") not in getattr(manager, "stages", manager.STAGES):
            rejected.append({"task_id": task_id, "reason": "unknown_stage"})
            continue
        collected = collect_calculation_results(manager, structure_id, result, ledger_path=None)
        _record_cost_observation(current, result, manager, structure_id)
        _record_final_frame_error(current, result, manager, structure_id, final_frame_mlip_evaluator)
        processed.append(task_id)
        accepted.append(collected)
        phase_record = collected.get("phase_record")
        if phase_record and phase_record["record_id"] not in {
            row.get("record_id") for row in current["phase_records"]
        }:
            current["phase_records"].append(phase_record)
            feedback_rows.append({**result, "phase_record": phase_record})

    current, _ = ensure_phase_identification(
        current, manager, phase_references=phase_references,
        cache_path=phase_identification_cache_path,
    )
    refresh_identified_phases(current)
    selected_model = active_model_version or current.get("active_model_version")
    if selected_model is not None:
        current["active_model_version"] = selected_model
    if current.get("phase_records"):
        previous = deepcopy(current["phase_diagrams"])
        previous_models = current["phase_diagrams_by_model"]
        output = update_phase_diagram(
            current["phase_records"], output_directory=phase_diagram_directory,
            parent_versions={**{name: row.get("version") for name, row in previous.items()},
                             **{f"mlip:{name}": row.get("version") for name, row in previous_models.items()}},
            active_model_version=active_model_version or current.get("active_model_version"),
        )
        diagrams = output["diagrams"]
        current["phase_diagrams_by_model"].update(output["mlip_by_version"])
        current["phase_diagrams"] = diagrams
        if feedback_rows:
            _record_convergence_round(current, previous, diagrams)
            _record_rewards(current, previous, diagrams, feedback_rows)
    current["coverage"] = coverage(manager.data, manager.stages, manager.stage_labels)
    current["agent_state_summary"] = summarize_agent_state(current)
    if ledger_path is not None and accepted:
        manager.save(ledger_path)
    return {
        "state": current, "recorded": accepted, "rejected": rejected,
        "feedback_count": len(feedback_rows),
    }


def _record_rewards(state, previous, diagrams, feedback_rows):
    for method in ("mlip", "dft"):
        relevant = [
            row for row in feedback_rows
            if row["phase_record"].get("energy_method") == method
            and (method != "mlip" or row["phase_record"].get("model_version") ==
                 (diagrams.get("mlip") or {}).get("model_version"))
        ]
        before, after = previous.get(method), diagrams.get(method)
        if (not relevant or not before or not after or after.get("status") != "completed"
                or before.get("energy_basis_id") != after.get("energy_basis_id")):
            continue
        reward = calculate_search_reward(
            before, after,
            batch_id="feedback:" + method + ":" + "+".join(sorted(row["task_id"] for row in relevant)),
            task_ids=[row["task_id"] for row in relevant],
            actual_cost=sum(_cost(row.get("actual_cost")) for row in relevant),
            reward_state=state["reward_states"].get(method),
        )
        reward.update({
            "energy_method": method,
            "energy_basis_id": after.get("energy_basis_id"),
            "previous_version": before.get("version"),
            "current_version": after.get("version"),
        })
        state["reward_states"][method] = reward.pop("state")
        state["rewards"].append(reward)


def _cost(value):
    if isinstance(value, dict):
        return float(value.get("value", value.get("relative_cost", 0)) or 0)
    return float(value or 0)


def _record_cost_observation(state, result, manager, structure_id):
    raw_actual = result.get("actual_cost")
    actual = _cost(raw_actual) if raw_actual is not None else None
    estimated = result.get("estimated_cost")
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == result.get("task_id")), {})
    planned = float(task.get("planned_relative_cost", 0) or 0)
    record = manager.data["structures"].get(structure_id) or {}
    atoms = (record.get("metadata") or {}).get("atom_count") or result.get("outputs", {}).get("atom_count")
    state.setdefault("cost_history", []).append({"task_id": result.get("task_id"),
        "stage": result.get("stage"), "atom_count": atoms, "planned_cost": planned,
        "actual_cost": actual, "estimated_cost": estimated,
        "actual_cost_known": actual is not None,
        "actual_gpu_core_hours": result.get("actual_gpu_core_hours"),
        "status": result.get("status")})


def _record_final_frame_error(state, result, manager, structure_id, evaluator=None):
    if result.get("status") != "completed" or result.get("stage") not in {"dft_single_point", "dft_relax"}:
        return
    outputs = result.get("outputs") or {}
    dft_energy = outputs.get("energy")
    mlip_energy = outputs.get("final_frame_mlip_energy")
    if mlip_energy is None and callable(evaluator):
        evaluated = evaluator(result=deepcopy(result), structure_id=structure_id, manager=manager)
        mlip_energy = evaluated.get("energy") if isinstance(evaluated, dict) else evaluated
    if dft_energy is None or mlip_energy is None:
        return
    record = manager.data["structures"].get(structure_id) or {}
    atoms = outputs.get("atom_count") or (record.get("metadata") or {}).get("atom_count")
    if not atoms:
        composition = record.get("composition") or {}
        atoms = sum(float(value) for value in composition.values())
    if not atoms:
        return
    state.setdefault("final_frame_dft_errors", []).append({"task_id": result.get("task_id"), "structure_id": structure_id, "error_ev_per_atom": abs(float(dft_energy) - float(mlip_energy)) / float(atoms), "comparison": "mlip_vs_dft_on_final_stable_structure", "dft_energy": float(dft_energy), "mlip_energy": float(mlip_energy)})


def _record_convergence_round(state, previous, current):
    method = "dft" if (current.get("dft") or {}).get("entries") else "mlip"
    before, after = previous.get(method) or {}, current.get(method) or {}
    if before.get("energy_basis_id") != after.get("energy_basis_id"):
        return
    if not before.get("entries") or not after.get("entries"):
        return
    old = {row.get("structure_id"): row for row in before["entries"]}
    new = {row.get("structure_id"): row for row in after["entries"]}
    common = set(old) & set(new)
    hull_change = max((abs(float(old[key]["ehull"]) - float(new[key]["ehull"])) for key in common), default=None)
    old_ground = {key for key, row in old.items() if row.get("is_stable")}
    new_ground = {key for key, row in new.items() if row.get("is_stable")}
    evidence = {"method": method, "previous_version": before.get("version"),
                "current_version": after.get("version"), "hull_change": hull_change,
                "ground_state_ids": sorted(new_ground),
                "ground_state_unchanged": old_ground == new_ground}
    state.setdefault("convergence_history", []).append(evidence)
    # Attach evidence to an existing model-update epoch; never create an epoch
    # from an MC/result sub-round under the same model.
    active_version = state.get("active_model_version")
    epoch = next((row for row in reversed(state.get("model_update_epochs") or [])
                  if row.get("model_version") == active_version), None)
    if epoch is not None:
        epoch.update({"hull_change": hull_change,
                      "ground_state_unchanged": old_ground == new_ground,
                      "hull_version": after.get("version"),
                      "convergence_evidence_method": method})
