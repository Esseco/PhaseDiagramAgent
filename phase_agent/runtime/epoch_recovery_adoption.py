"""Explicitly adopt the current epoch recovery policy for a settled legacy cycle."""
from copy import deepcopy
from pathlib import Path
import json
from filelock import FileLock


def adopt_state(state, expected_round_digest):
    from phase_agent.agents.reliability.extrapolation_round import (
        assess_current_rounds, recovery_dft_failure_settled,
    )
    from phase_agent.evidence.dft_failure_policy import unreviewed_failure_batches
    from phase_agent.evidence.extrapolation_policy import recovery_lineage
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.evidence.parent_first_recovery import valid_parent_first_cycle

    cycle = state.get("extrapolation_recovery_cycle") or {}
    if cycle.get("policy_version") == "epoch_refresh_v1":
        if cycle.get("round_evidence_digest") != expected_round_digest:
            raise ValueError("recovery_policy_round_mismatch")
        return state, {"status": "already_adopted", "progress": recovery_label_facts(state)}
    parent_first = (cycle.get("recovery_route") == "parent_first_perturbation"
                    or (not cycle.get("recovery_route") and bool(cycle.get("batch_id"))))
    if (cycle.get("policy_version") or cycle.get("scope_id") != "epoch_refresh"
            or cycle.get("status") != "awaiting_reviewed_dft_labels"
            or cycle.get("invalid_model_version") != state.get("active_model_version")):
        raise ValueError("recovery_policy_legacy_cycle_invalid")
    if parent_first:
        if not valid_parent_first_cycle(state):
            raise ValueError("recovery_policy_legacy_cycle_invalid")
    else:
        from phase_agent.tools.workflows.validate_dft_agent_decisions import _is_direct_recovery_selection

        ids = {row.get("candidate_id") for row in cycle.get("selected") or []}
        receipt = (state.get("direct_sampling_results") or {}).get("epoch_refresh") or {}
        if (cycle.get("recovery_route") not in {"extrapolation", "perturbation"}
                or receipt.get("round_evidence_digest") != expected_round_digest
                or not _is_direct_recovery_selection(state, ids)):
            raise ValueError("recovery_policy_legacy_cycle_invalid")
    if unreviewed_failure_batches(state):
        raise ValueError("recovery_policy_failure_analysis_required")
    facts = next((r for r in assess_current_rounds(state)["rounds"]
                  if r.get("scope_id") == "epoch_refresh"), {})
    if (facts.get("evidence_digest") != expected_round_digest
            or facts.get("direct_sampling_required") is not True):
        raise ValueError("recovery_policy_round_mismatch")
    result = deepcopy(state)
    adopted = result["extrapolation_recovery_cycle"]
    adopted["policy_version"] = "epoch_refresh_v1"
    adopted["round_evidence_digest"] = expected_round_digest
    adopted.update(recovery_lineage(None, adopted["invalid_model_version"],
                                    expected_round_digest,
                                    model_epoch(result, adopted["invalid_model_version"])))
    progress = recovery_label_facts(result)
    if (not progress.get("all_returned")
            or (parent_first and not recovery_dft_failure_settled(result, adopted))
            or (parent_first and (progress.get("status") != "waiting"
                                   or progress.get("threshold_met") is not False))
            or (not parent_first and (progress.get("status") != "complete"
                                       or progress.get("threshold_met") is not True))):
        raise ValueError("recovery_policy_results_not_settled_or_sufficient")
    if parent_first:
        batch = result["parent_first_recovery_batches"][adopted["batch_id"]]
        batch["status"] = "results_received"
    adopted["legacy_policy_adoption"] = {
        "decision": "exclude_failed_without_retry",
        "round_evidence_digest": expected_round_digest,
        "selected_count": progress["selected_count"],
        "labelled_count": progress["labelled_count"],
        "failed_task_count": progress["failed_count"],
        "basis": "explicit_user_instruction",
    }
    return result, {"status": "adopted", "progress": progress}


def adopt(control, expected_round_digest):
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    path = Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=30):
        before = checksum(path)
        state = json.loads(path.read_text(encoding="utf8"))
        updated, receipt = adopt_state(state, expected_round_digest)
        if receipt["status"] == "adopted":
            commit_result_recovery(path, updated, {}, expected_hashes={str(path): before},
                                   evidence={"operation": "adopt_epoch_recovery_policy",
                                             "round_evidence_digest": expected_round_digest,
                                             "failed_disposition": "exclude_without_retry"})
        control._codex_context_cache = None
        return receipt

