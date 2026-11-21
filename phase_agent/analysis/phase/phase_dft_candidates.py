"""Supplement MC candidates with bound thermodynamically relevant review geometries."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from phase_agent.analysis.phase.phase_quality_reports import (
    candidate_quality_evidence,
    get_phase_quality_report,
)


def _energy_source_version(row, snapshot):
    """Report the model that supplied this frame's energy, not the hull owner."""
    source = row.get("source_version")
    if source:
        return source
    versions = snapshot.get("source_model_versions") or []
    if snapshot.get("mixed_model_versions") is True or len(set(versions)) > 1:
        return None
    return snapshot.get("model_version")


def phase_review_candidates(state, candidates, *, report=None):
    """Expose evidence only; no new model calls, selection, task creation or exclusion."""
    from phase_agent.evidence.model_diagram_scope import selection_state

    state = selection_state(state)
    snapshot = state["phase_diagrams"]["mlip"]
    model = state.get("active_model_version")
    if snapshot.get("status") != "completed" or snapshot.get("model_version") != model:
        return refresh_failure_candidates(state, candidates)
    report = report if report is not None else get_phase_quality_report(state)
    if report.get("model_version") != model or report.get("hull_version") != snapshot.get(
        "version"
    ):
        return refresh_failure_candidates(state, bind_ground_state_frames(state, candidates))
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
    # A force-triggered refresh keeps its original-frame prediction separately
    # from the final relaxed frame. Only reuse an explicitly bound QBC.
    initial_qbc = {}
    for task in state.get("tasks") or []:
        followup = (task.get("outputs") or {}).get("refresh_force_followup") or {}
        initial = followup.get("initial_prediction") or {}
        digest = initial.get("structure_checksum")
        qbc = initial.get("qbc") or {}
        if (
            task.get("status") == "completed"
            and task.get("model_version") == model
            and initial.get("status") == "completed"
            and initial.get("single_point_completed") is True
            and initial.get("mlip_version") == model
            and digest
            and digest == task.get("input_structure_sha256")
            and digest == followup.get("input_structure_sha256")
            and qbc.get("status") == "completed"
        ):
            initial_qbc[digest] = (task, qbc)
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
        qbc = (source.get("outputs") or {}).get("qbc")
        if (not qbc or qbc.get("status") != "completed") and digest in initial_qbc:
            source, qbc = initial_qbc[digest]
        result.append(
            {
                "candidate_id": sid,
                "structure_id": sid,
                "branch_id": source.get("branch_id"),
                "source_task_id": source.get("task_id"),
                "candidate_origin": "current_phase_review" if digest in eligible else "D_reference",
                "structure_path": raw,
                "structure_sha256": digest,
                "model_version": _energy_source_version(row, snapshot),
                "energy_source_version": _energy_source_version(row, snapshot),
                "composition": deepcopy(row.get("composition")),
                "atom_count": sum((row.get("composition") or {}).values()),
                "phase": row.get("phase"),
                "x_Na_per_O2": row.get("x_Na_per_O2"),
                "predicted_Ehull": row["ehull"],
                "ehull_unit": row["ehull_unit"],
                "phase_diagram_version": snapshot["version"],
                "qbc": deepcopy(qbc or {"status": "not_available_exact_current_geometry"}),
                "qbc_frame_sha256": digest if qbc and qbc.get("status") == "completed" else None,
                "qbc_model_version": model if qbc and qbc.get("status") == "completed" else None,
            }
        )
        ids.add(sid)
    bound = bind_ground_state_frames(state, result)
    _verify_candidate_qbc(state, bound, model)
    return candidate_quality_evidence(
        state, refresh_failure_candidates(state, bound), report=report
    )



def _verify_candidate_qbc(state, candidates, model):
    """Bind completed QBC to a completed current-model task and exact frame."""
    from collections import defaultdict

    evidence = defaultdict(list)
    for task in state.get("tasks") or []:
        if task.get("status") != "completed" or task.get("model_version") != model or task.get("checks_passed") is False:
            continue
        output = task.get("outputs") or {}
        if (output.get("extrapolation_monitor") or {}).get("extrapolation_failure") is True:
            continue
        qbc = output.get("qbc") or {}
        digest = output.get("structure_checksum")
        if output.get("mlip_version") in (None, model) and digest and qbc.get("status") == "completed":
            evidence[digest].append((task.get("task_id"), qbc))
        followup = output.get("refresh_force_followup") or {}
        initial = followup.get("initial_prediction") or {}
        digest = initial.get("structure_checksum")
        qbc = initial.get("qbc") or {}
        if (
            initial.get("status") == "completed"
            and initial.get("single_point_completed") is True
            and initial.get("mlip_version") == model
            and digest
            and digest == task.get("input_structure_sha256")
            and digest == followup.get("input_structure_sha256")
            and qbc.get("status") == "completed"
        ):
            evidence[digest].append((task.get("task_id"), qbc))

    for candidate in candidates:
        if (candidate.get("qbc") or {}).get("status") != "completed":
            continue
        digest = candidate.get("structure_sha256")
        matches = evidence.get(digest, [])
        if matches and all(_qbc_equivalent(qbc, matches[0][1]) for _, qbc in matches):
            candidate["qbc"] = deepcopy(matches[0][1])
            candidate["qbc_frame_sha256"] = digest
            candidate["qbc_model_version"] = model
            candidate["qbc_source_task_id"] = matches[0][0]
        else:
            candidate["qbc"] = {"status": "not_available_exact_current_geometry"}
            for key in ("qbc_frame_sha256", "qbc_model_version", "qbc_source_task_id"):
                candidate.pop(key, None)


def refresh_failure_candidates(state, candidates):
    """Keep failure frames out of the DFT candidate list; raw evidence remains saved."""
    return [row for row in candidates if row.get("extrapolation_failure") is not True]


def extrapolation_dft_candidates(state, candidates):
    """Expose exact recovered failure frames for review; never infer hull energy."""
    result = deepcopy(candidates)
    ids = {row.get("candidate_id") or row.get("structure_id") for row in result}
    model = state.get("active_model_version")
    from phase_agent.evidence.endpoint_supplement import supplement_facts

    supplement = supplement_facts(state)
    for task in state.get("tasks") or []:
        evidence = task.get("extrapolation_review") or {}
        outputs = task.get("outputs") or {}
        monitor = outputs.get("extrapolation_monitor") or {}
        if task.get("model_version") == model:
            for index, failure in enumerate(outputs.get("extrapolation_failures") or []):
                if (failure.get("extrapolation_monitor") or {}).get(
                    "trigger"
                ) == "historical_final" and (
                    not supplement
                    or supplement.get("invalid")
                    or task.get("task_id") not in supplement["task_ids"]
                ):
                    continue
                path = failure.get("structure_path")
                digest = failure.get("structure_checksum")
                cid = f"EXTRAP-{task.get('task_id')}-{index}"
                if not path or not digest or not task.get("structure_id") or cid in ids:
                    continue
                try:
                    from pymatgen.core import Structure

                    if sha256(Path(path).read_bytes()).hexdigest() != digest:
                        continue
                    structure = Structure.from_file(path)
                except (OSError, ValueError):
                    continue
                result.append(
                    {
                        "candidate_id": cid,
                        "structure_id": task["structure_id"],
                        "source_task_id": task.get("task_id"),
                        "branch_id": task.get("branch_id"),
                        "candidate_origin": "extrapolation_failure",
                        "extrapolation_failure": True,
                        "extrapolation_evidence": deepcopy(
                            failure.get("extrapolation_monitor") or {}
                        ),
                        "structure_path": path,
                        "structure_sha256": digest,
                        "model_version": model,
                        "composition": structure.composition.as_dict(),
                        "atom_count": len(structure),
                        "predicted_Ehull": None,
                        "qbc": {"status": "not_available_exact_current_geometry"},
                        "input_frame_kind": "recovered_extrapolation_failure_frame",
                    }
                )
                ids.add(cid)
        if (
            outputs.get("extrapolation_failures")
            or task.get("model_version") != model
            or not (
                evidence.get("status") == "extrapolation_failure"
                or monitor.get("extrapolation_failure") is True
            )
        ):
            continue
        tid, sid = task.get("task_id"), task.get("structure_id")
        path, digest = outputs.get("structure_path"), outputs.get("structure_checksum")
        if not tid or not sid or not path or not digest:
            continue
        try:
            if sha256(Path(path).read_bytes()).hexdigest() != digest:
                continue
        except OSError:
            continue
        cid = "EXTRAP-" + tid
        if cid in ids:
            continue
        composition = deepcopy(outputs.get("composition") or {})
        result.append(
            {
                "candidate_id": cid,
                "structure_id": sid,
                "source_task_id": tid,
                "branch_id": task.get("branch_id"),
                "candidate_origin": "extrapolation_failure",
                "extrapolation_failure": True,
                "extrapolation_evidence": deepcopy(evidence or monitor),
                "structure_path": path,
                "structure_sha256": digest,
                "model_version": model,
                "composition": composition,
                "atom_count": outputs.get("atom_count") or sum(composition.values()),
                "phase": outputs.get("actual_phase") or outputs.get("source_phase"),
                "predicted_Ehull": None,
                "qbc": deepcopy(
                    outputs.get("qbc") or {"status": "not_available_exact_current_geometry"}
                ),
                "input_frame_kind": "recovered_extrapolation_failure_frame",
                "optimizer_converged": outputs.get("optimizer_converged"),
            }
        )
        ids.add(cid)
    return result


def _qbc_equivalent(left, right):
    """Allow only numerical roundoff in duplicate same-frame QBC results."""
    from math import isclose

    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _qbc_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _qbc_equivalent(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        return isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def bind_ground_state_frames(state, candidates):
    """Bind each current hull GS to an actual, verified GS frame, never a newer frame."""
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("status") != "completed" or snapshot.get("model_version") != state.get(
        "active_model_version"
    ):
        return deepcopy(candidates)
    from collections import defaultdict

    groups = defaultdict(list)
    qbc_by_frame = defaultdict(list)
    model = state.get("active_model_version")
    for task in state.get("tasks") or []:
        if (
            task.get("status") != "completed"
            or task.get("model_version") != model
            or task.get("checks_passed") is False
        ):
            continue
        output = task.get("outputs") or {}
        if (
            output.get("mlip_version") not in (None, model)
            or (output.get("extrapolation_monitor") or {}).get("extrapolation_failure") is True
        ):
            continue
        qbc = output.get("qbc") or {}
        digest = output.get("structure_checksum")
        if digest and qbc.get("status") == "completed":
            qbc_by_frame[digest].append((task.get("task_id"), qbc))
    for row in snapshot.get("entries") or []:
        if row.get("is_stable"):
            sid = row.get("structure_id")
            if not sid:
                raise ValueError("ground_state_missing_structure_id")
            path = Path(row.get("structure_path") or "")
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row.get(
                "structure_sha256"
            ):
                raise ValueError("ground_state_frame_invalid:" + sid)
            groups[(sid, row["structure_sha256"])].append(row)
    result = deepcopy(candidates)
    indices = {
        row.get("candidate_id") or row.get("structure_id"): i for i, row in enumerate(result)
    }
    counts = defaultdict(int)
    for sid, digest in groups:
        counts[sid] += 1
    # A legacy single-ID candidate may retain a stale GS marker after a hull
    # gains multiple frames; only the frame-qualified entries carry GS status.
    for item in result:
        sid = item.get("structure_id")
        if counts.get(sid, 0) > 1 and item.get("candidate_id") == sid:
            item.pop("ground_state_record_ids", None)
            item.pop("ground_state_frame_sha256", None)
            if item.get("candidate_origin") == "current_hull_ground_state":
                item["candidate_origin"] = "prior_geometry"
    for (sid, digest), rows in sorted(groups.items()):
        row = min(
            rows, key=lambda r: (r.get("energy_per_O2", float("inf")), str(r.get("record_id")))
        )
        cid = sid if counts[sid] == 1 else sid + ":frame:" + digest
        index = indices.get(cid)
        source = next(
            (
                item
                for item in result
                if item.get("structure_id") == sid and item.get("structure_sha256") == digest
            ),
            None,
        )
        old = result[index] if index is not None else (source or {})
        same_frame = old.get("structure_sha256") == digest
        item = deepcopy(old)
        item.update(
            {
                "candidate_id": cid,
                "structure_id": sid,
                "candidate_origin": "current_hull_ground_state",
                "structure_path": row["structure_path"],
                "structure_sha256": digest,
                "ground_state_record_ids": [r.get("record_id") for r in rows],
                "ground_state_frame_sha256": digest,
                "model_version": _energy_source_version(row, snapshot),
                "energy_source_version": _energy_source_version(row, snapshot),
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
            item.pop("qbc_frame_sha256", None)
            item.pop("qbc_model_version", None)
            item.pop("qbc_source_task_id", None)
        if (item.get("qbc") or {}).get("status") != "completed":
            matches = qbc_by_frame.get(digest, [])
            if matches and all(_qbc_equivalent(qbc, matches[0][1]) for _, qbc in matches):
                item["qbc"] = deepcopy(matches[0][1])
                item["qbc_frame_sha256"] = digest
                item["qbc_model_version"] = model
                item["qbc_source_task_id"] = matches[0][0]
        if index is not None:
            result[index] = item
        else:
            indices[cid] = len(result)
            result.append(item)
    return result
