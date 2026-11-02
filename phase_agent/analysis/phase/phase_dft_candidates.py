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
        return bind_ground_state_frames(state, candidates)
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
    return candidate_quality_evidence(state, bind_ground_state_frames(state, result), report=report)


def bind_ground_state_frames(state, candidates):
    """Bind each current hull GS to an actual, verified GS frame, never a newer frame."""
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("status") != "completed" or snapshot.get("model_version") != state.get(
        "active_model_version"
    ):
        return deepcopy(candidates)
    from collections import defaultdict
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher, SpeciesComparator

    groups = defaultdict(list)
    for row in snapshot.get("entries") or []:
        if row.get("is_stable"):
            if not row.get("structure_id"):
                raise ValueError("ground_state_missing_structure_id")
            groups[row["structure_id"]].append(row)
    result = deepcopy(candidates)
    indices = {
        row.get("candidate_id") or row.get("structure_id"): i for i, row in enumerate(result)
    }
    matcher = StructureMatcher(
        ltol=0.001,
        stol=0.001,
        angle_tol=0.1,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        comparator=SpeciesComparator(),
    )
    for sid, rows in groups.items():
        structures = []
        for row in rows:
            path = Path(row.get("structure_path") or "")
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row.get(
                "structure_sha256"
            ):
                raise ValueError("ground_state_frame_invalid:" + sid)
            structures.append(Structure.from_file(str(path)))
        if any(
            row.get("phase") != rows[0].get("phase")
            or len(structures[0]) != len(structure)
            or not matcher.fit(structures[0], structure)
            for row, structure in zip(rows[1:], structures[1:])
        ):
            raise ValueError("ground_state_inequivalent:" + sid)
        row = min(
            rows, key=lambda r: (r.get("energy_per_O2", float("inf")), str(r.get("record_id")))
        )
        old = result[indices[sid]] if sid in indices else {}
        same_frame = old.get("structure_sha256") == row["structure_sha256"]
        if not same_frame and old.get("structure_path"):
            old_path = Path(old["structure_path"])
            same_frame = (
                old_path.is_file()
                and sha256(old_path.read_bytes()).hexdigest() == row["structure_sha256"]
            )
        item = deepcopy(old)
        item.update(
            {
                "candidate_id": sid,
                "structure_id": sid,
                "candidate_origin": "current_hull_ground_state",
                "structure_path": row["structure_path"],
                "structure_sha256": row["structure_sha256"],
                "ground_state_record_ids": [r.get("record_id") for r in rows],
                "ground_state_frame_sha256": row["structure_sha256"],
                "model_version": snapshot["model_version"],
                "composition": deepcopy(row["composition"]),
                "atom_count": sum(row["composition"].values()),
                "phase": row["phase"],
                "x_Na_per_O2": row["x_Na_per_O2"],
                "predicted_Ehull": row["ehull"],
                "ehull_unit": row["ehull_unit"],
                "phase_diagram_version": snapshot["version"],
            }
        )
        if not same_frame:
            item["qbc"] = {"status": "not_available_exact_current_geometry"}
            item["geometry_optimization_evidence"] = None
            item["source_search_task_id"] = None
        if sid in indices:
            result[indices[sid]] = item
        else:
            result.append(item)
    return result
