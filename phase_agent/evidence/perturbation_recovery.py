"""Verified parent structures for the perturbation recovery option."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from phase_agent.tools.workflows.generation_artifacts import determinant


def _normal_frame_digests(state):
    """Only explicitly monitored normal frames count; missing evidence is unknown."""
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    normal = {}
    failures = set()
    for task in tasks.values():
        output = task.get("outputs") or {}
        monitor = output.get("extrapolation_monitor") or {}
        score = monitor.get("final_feature_difference")
        digest = output.get("structure_checksum")
        if digest and (monitor.get("extrapolation_failure") is True
                       or (task.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"):
            failures.add(digest)
        if (task.get("status") == "completed" and monitor.get("extrapolation_failure") is False
            and isinstance(monitor.get("failures"), list) and not monitor["failures"]
            and isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and 0 <= score <= 1 and digest):
            normal[digest] = "final_monitor"
    if state.get("extrapolation_review_summary"):
        from phase_agent.evidence.extrapolation_review import extrapolation_review

        review = extrapolation_review(state)
        if review.get("status") != "endpoint_classified_incomplete":
            raise ValueError("perturbation_endpoint_evidence_invalid")
        flagged = next((t for t in tasks.values()
                        if (t.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"), None)
        if not flagged:
            raise ValueError("perturbation_endpoint_evidence_invalid")
        source = Path(flagged["extrapolation_review"]["source"])
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                task = tasks.get(row.get("task_id")) or {}
                score = float(row["score"])
                if row.get("classification") == "extrapolation_failure" and task.get("input_structure_sha256"):
                    failures.add(task["input_structure_sha256"])
                if (row.get("classification") == "below_threshold" and not row.get("error")
                    and math.isfinite(score) and 0 <= score <= 1
                    and task.get("model_version") == state.get("active_model_version")
                    and task.get("input_structure_sha256")):
                    normal.setdefault(task["input_structure_sha256"], "initial_final_feature_difference")
    supplement = state.get("endpoint_supplement") or {}
    if supplement:
        from phase_agent.evidence.endpoint_supplement import supplement_facts
        facts = supplement_facts(state)
        if not facts or facts.get("invalid"):
            raise ValueError("perturbation_endpoint_evidence_invalid")
        csv_path = Path(supplement["directory"]) / "evidence.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                task = tasks.get(row.get("task_id")) or {}
                score = float(row["score"])
                if (row.get("classification") == "extrapolation_failure"
                    and row.get("initial_sha256") == task.get("input_structure_sha256")):
                    failures.add(row["initial_sha256"])
                if (row.get("classification") == "below_threshold" and not row.get("error")
                    and math.isfinite(score) and 0 <= score <= 1
                    and row.get("task_id") in facts["task_ids"]
                    and row.get("initial_sha256") == task.get("input_structure_sha256")):
                    normal.setdefault(row["initial_sha256"], "initial_final_feature_difference")
    return {digest: basis for digest, basis in normal.items() if digest not in failures}


def build_perturbation_recovery_pool(state, ledger, *, max_det_H):
    """Expose verified candidates; Strategic selects scope and coverage, never this function."""
    if type(max_det_H) is not int or not 1 <= max_det_H <= 6:
        raise ValueError("perturbation_max_det_H_invalid")
    cycle = state.get("extrapolation_recovery_cycle") or {}
    refresh = state.get("model_refresh") or {}
    # A failed refresh must use its last valid pre-refresh hull even before
    # quarantine creates a recovery cycle. Never select parents from that failed hull.
    from phase_agent.agents.reliability.extrapolation_round import required_direct_round
    previous = (refresh.get("old_diagram") or {}) if required_direct_round(state) else {}
    if previous and previous.get("model_version") != refresh.get("old_model_version"):
        raise ValueError("perturbation_hull_binding_invalid")
    snapshot = cycle.get("baseline_diagram") or previous or (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("status") != "completed" or not snapshot.get("version"):
        raise ValueError("perturbation_valid_hull_required")
    if cycle and (snapshot.get("version") != cycle.get("baseline_hull_version")
                  or snapshot.get("model_version") != cycle.get("baseline_model_version")):
        raise ValueError("perturbation_hull_binding_invalid")
    normal = _normal_frame_digests(state)
    records = {r.get("record_id"): r for r in state.get("phase_records") or []}
    structures, branches = ledger.get("structures") or {}, ledger.get("branches") or {}
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    rows = []
    for entry in snapshot.get("entries") or []:
        digest = entry.get("structure_sha256")
        if digest not in normal or entry.get("phase_identification_status") != "identified":
            continue
        record = records.get(entry.get("record_id")) or {}
        task = tasks.get(record.get("source_task_id")) or {}
        sid = entry.get("structure_id")
        branch_id = task.get("branch_id") or (structures.get(sid) or {}).get("branch_id")
        branch = branches.get(branch_id) or {}
        det = determinant(branch.get("H"))
        if type(det) is not int or not 1 <= det <= max_det_H:
            continue
        if record.get("structure_sha256") != digest or record.get("structure_id") != sid:
            continue
        path = Path(entry.get("structure_path") or "")
        try:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                continue
        except OSError:
            continue
        rows.append({"record_id": entry["record_id"], "structure_id": sid, "branch_id": branch_id,
                     "structure_path": str(path), "structure_sha256": digest, "phase": entry.get("phase"),
                     "x_Na_per_O2": entry.get("x_Na_per_O2"), "H": branch["H"], "det_H": det,
                     "hull_version": snapshot["version"], "normal_evidence": normal[digest]})
    return {"hull_version": snapshot["version"], "model_version": snapshot.get("model_version"),
            "max_det_H": max_det_H, "candidates": rows,
            "instruction": "由 Strategic 自主按相、Na含量和扩胞广覆盖选择；缺失监控不等于外推正常。"}
