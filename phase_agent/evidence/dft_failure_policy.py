"""DFT failure review facts and read-only Simulation diagnosis."""

import hashlib
import json

from phase_agent.evidence.dft_round_status import dft_recovery_rounds


def _is_epoch_recovery_group(state, rows, expected_count):
    """Keep recovery failure summaries out of the separate DFT review gate."""
    cycle = state.get("extrapolation_recovery_cycle") or {}
    if cycle.get("policy_version") != "epoch_refresh_v1" or not rows or len(rows) != expected_count:
        return False
    attempts = [*state.get("extrapolation_recovery_history", []), cycle]
    from phase_agent.evidence.extrapolation_recovery import _matches_recovery_frame
    bindings = {}
    for row in attempts:
        if (row.get("policy_version") == "epoch_refresh_v1"
            and row.get("scope_id") == cycle.get("scope_id")
            and row.get("invalid_model_version") == cycle.get("invalid_model_version")):
            for item in row.get("selected") or []:
                bindings.setdefault(item.get("structure_sha256"), set()).add(item.get("candidate_id"))
    if not bindings or None in bindings:
        return False
    for task in rows:
        if (task.get("stage") not in {"dft_single_point", "dft_relax"}
            or not _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version"))):
            return False
    return True


def completed_failure_batches(state, reconciled):
    """Bound a DFT batch's failure rate; unknown work never counts as failure.

    A finished batch uses returned calculations as denominator. While work is
    outstanding, trigger only when the threshold is already unavoidable even
    if every outstanding calculation succeeds. This never cancels running work.
    """
    groups = {}
    for row in dft_recovery_rounds(state):
        scope = row["scope"]
        identity = {k: scope.get(k) for k in
                    ("model_version", "upload_operation_id", "parent_relax_round", "retry_index")}
        if not identity["upload_operation_id"]:
            identity["batch_id"] = scope.get("batch_id")
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(key, {"scope": identity, "expected": 0, "cancelled": 0,
                                        "ids": set(), "pending": set(), "waived": set()})
        group["ids"].update(row["recovered_task_ids"])
        group["ids"].update(row["pending_task_ids"])
        group["ids"].update(row["waived_task_ids"])
        group["ids"].update(row["cancelled_task_ids"])
        group["pending"].update(row["pending_task_ids"])
        group["waived"].update(row["waived_task_ids"])
        group["expected"] += row["expected_tasks"]
        group["cancelled"] += row["cancelled_tasks"]
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    active = state.get("active_model_version")
    result = []
    for key, group in groups.items():
        if active and group["scope"].get("model_version") and group["scope"]["model_version"] != active:
            continue
        rows = [tasks[tid] for tid in group["ids"] if tid in tasks]
        returned = [t for t in rows if t.get("status") in {"completed", "failed", "timeout"}]
        failures = [t for t in returned if t.get("status") in {"failed", "timeout"}]
        denominator = group["expected"] - group["cancelled"]
        # Include unknown/waived tasks as potential future successes until resolved.
        if denominator <= 0 or len(failures) * 5 < denominator:
            continue
        evidence = [{"task_id": t["task_id"], "stage": t.get("stage"),
                     "error": t.get("error") or t.get("failure_reason"),
                     "result_path": t.get("result_path")} for t in sorted(failures, key=lambda x: x["task_id"])]
        identity = {"batch_key": key, "failed_task_ids": [r["task_id"] for r in evidence],
                    "failure_evidence": evidence}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result.append({"batch_key": key, "scope": group["scope"], "returned_count": len(returned),
                       "failed_count": len(failures), "expected_count": group["expected"],
                       "eligible_count": denominator, "pending_count": max(0, denominator - len(returned)),
                       "waived_count": len(group["waived"]), "threshold": 0.2,
                       "threshold_basis": "completed_batch" if len(returned) == denominator else "guaranteed_lower_bound",
                       "failure_evidence_digest": digest, "failure_evidence": evidence})
    return result

def valid_failure_diagnosis(report):
    from phase_agent.agents.simulation.contracts import SimulationDiagnosis
    try:
        parsed = SimulationDiagnosis.model_validate(report.get("diagnosis"))
        return "execution.operation_outcome" in parsed.evidence
    except (ValueError, TypeError, AttributeError):
        return False


def unreviewed_failure_batches(state):
    """Fail closed even when an action bypasses the ordinary lifecycle wait gate."""
    reviews = state.get("dft_failure_reviews") or []
    checked = {row.get("batch", {}).get("failure_evidence_digest") for row in reviews
               if isinstance(row, dict) and valid_failure_diagnosis(row)}
    return [row for row in completed_failure_batches(state, [])
            if row["failure_evidence_digest"] not in checked]


def failure_guard_facts(state):
    """Read-only current failure threshold and diagnostic status."""
    batches = completed_failure_batches(state, [])
    if not batches:
        return {"status": "clear", "threshold": 0.2, "batches": []}
    batch_digests = {b["failure_evidence_digest"] for b in batches}
    reports = [r for r in (state.get("dft_failure_reviews") or [])
               if isinstance(r, dict) and valid_failure_diagnosis(r)
               and r.get("batch", {}).get("failure_evidence_digest") in batch_digests]
    reviewed = {r["batch"]["failure_evidence_digest"] for r in reports}
    pending = batch_digests - reviewed
    return {"status": "awaiting_analysis" if pending else "analyzed", "threshold": 0.2,
            "batches": batches, "reports": reports,
            "instruction": "DFT失败达到20%，先核查实际原因和风险；保留成功结果，未回收不算失败，不自动重试。"}



def diagnose_failure_batch(batch, client, *, extra_instruction=""):
    """Use the existing Simulation contract; never execute or retry a task."""
    from phase_agent.agents.simulation.agent import client_for
    from phase_agent.agents.simulation.contracts import SimulationDiagnosis
    from phase_agent.agents.contracts import model_content

    observations = {
        "execution.status": "failed",
        "execution.error": "Selected DFT tasks failed; see recorded batch evidence",
        "execution.result.status": "failed",
        "execution.operation_outcome": batch,
    }
    request = {
        "mode": "simulation_diagnosis",
        "agent_role": "simulation",
        "allowed_tools": [],
        "observations": observations,
        "instruction": (
            "Diagnose the supplied DFT failure evidence only; cite exact observation keys. "
            "Distinguish infrastructure, input and computational failures when evidence permits; "
            "describe uncertainty and whether perturbation sampling merits Strategic review. "
            "Do not execute or retry. Return JSON in the requested schema." + extra_instruction
        ),
        "output_contracts": {"simulation_diagnosis": SimulationDiagnosis.model_json_schema()},
    }
    result = {"diagnosis_request": request}
    diagnostician = client_for(client)
    if diagnostician is None:
        result["diagnosis_unavailable"] = "simulation_client_unavailable"
        return result
    try:
        response = diagnostician(request)
        if isinstance(response, dict) and response.get("_llm_usage"):
            result["llm_usage"] = response["_llm_usage"]
        diagnosis = SimulationDiagnosis.model_validate(model_content(response)).model_dump()
        if "execution.operation_outcome" not in diagnosis["evidence"]:
            raise ValueError("dft_failure_analysis_missing_batch_evidence")
        result["diagnosis"] = diagnosis
    except Exception as error:
        result["diagnosis_unavailable"] = type(error).__name__
        if getattr(error, "llm_usage", None):
            result["llm_usage"] = error.llm_usage
    return result
