"""Supplement MC candidates with bound thermodynamically relevant review geometries."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from phase_agent.analysis.phase.phase_quality_reports import (
    candidate_quality_evidence,
    get_phase_quality_report,
)


def phase_review_candidates(state, candidates, *, report=None):
    """Expose evidence only; no new model calls, selection, task creation or exclusion."""
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    model = state.get("active_model_version")
    if snapshot.get("status") != "completed" or snapshot.get("model_version") != model:
        return deepcopy(candidates)
    report = report if report is not None else get_phase_quality_report(state)
    if report.get("model_version") != model or report.get("hull_version") != snapshot.get(
        "version"
    ):
        return deepcopy(candidates)
    eligible = {
        digest
        for row in report.get("structures") or []
        if row.get("quality_sampling_evidence_eligible")
        for digest in row.get("structure_sha256") or []
    }
    from phase_agent.analysis.phase.phase_dft_sampling import fixed_snapshot_sampling_evidence

    sampling = fixed_snapshot_sampling_evidence(snapshot, report)
    references = {
        path
        for key in eligible
        for path in (sampling.get(key) or {}).get("D_reference_structure_paths", [])
    }
    result = deepcopy(candidates)
    ids = {row.get("candidate_id") or row.get("structure_id") for row in result}
    for row in snapshot.get("entries") or []:
        sid, digest, raw = (
            row.get("structure_id"),
            row.get("structure_sha256"),
            row.get("structure_path"),
        )
        if (
            not sid
            or sid in ids
            or (digest not in eligible and raw not in references)
            or not raw
            or row.get("phase_identification_status") != "identified"
            or row.get("ehull_unit") != "eV/atom"
            or row.get("ehull") is None
        ):
            continue
        path = Path(raw)
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            continue
        matches = []
        for task in state.get("tasks") or []:
            outputs = task.get("outputs") or {}
            identity = (outputs.get("phase_identification") or {}).get(
                "structure_sha256"
            ) or outputs.get("structure_checksum")
            same = (
                identity == digest
                if identity
                else (outputs.get("structure_path") or outputs.get("final_structure_path")) == raw
            )
            if (
                task.get("model_version") == model
                and outputs.get("mlip_version") in (None, model)
                and task.get("status") == "completed"
                and same
            ):
                matches.append(task)
        source = next(
            (task for task in reversed(matches) if (task.get("outputs") or {}).get("qbc")),
            matches[-1] if matches else {},
        )
        result.append(
            {
                "candidate_id": sid,
                "structure_id": sid,
                "branch_id": source.get("branch_id"),
                "source_task_id": source.get("task_id"),
                "candidate_origin": "current_phase_review" if digest in eligible else "D_reference",
                "structure_path": raw,
                "structure_sha256": digest,
                "model_version": model,
                "composition": deepcopy(row.get("composition")),
                "atom_count": sum((row.get("composition") or {}).values()),
                "phase": row.get("phase"),
                "x_Na_per_O2": row.get("x_Na_per_O2"),
                "predicted_Ehull": row["ehull"],
                "ehull_unit": row["ehull_unit"],
                "phase_diagram_version": snapshot["version"],
                "qbc": deepcopy(
                    (source.get("outputs") or {}).get("qbc")
                    or {"status": "not_available_exact_current_geometry"}
                ),
            }
        )
        ids.add(sid)
    return candidate_quality_evidence(state, result, report=report)
