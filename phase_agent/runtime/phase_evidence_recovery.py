"""Recover selected, bound phase evidence without modifying original results."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


def prepare_bound_proofs(state, spec, epoch_root, task_ids, *, classify=None):
    """Reclassify only verified original result files using the frozen rule scope."""
    from pymatgen.core import Structure
    from phase_agent.analysis.phase.identify_result_phase import identify_result_phase
    from phase_agent.tools.remote.integrity import file_checksum, verify_result
    import json

    classify = classify or identify_result_phase
    root = Path(epoch_root).resolve()
    ids = list(task_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("phase_recovery_invalid_scope")
    rows = spec.get("tasks") or []
    if len({r.get("task_id") for r in rows}) != len(rows):
        raise ValueError("phase_recovery_duplicate_spec")
    by_spec = {r.get("task_id"): r for r in rows}
    tasks = state.get("tasks") or []
    counts = Counter(t.get("task_id") for t in tasks)
    by_task = {t.get("task_id"): t for t in tasks}
    manager = SimpleNamespace(boundary=spec["manager"]["boundary"], data=spec["manager"])
    references = {k: Structure.from_dict(v) for k, v in (spec.get("references") or {}).items()}
    proofs = {}
    scope_cache = {}
    for tid in ids:
        task, row = by_task.get(tid) or {}, by_spec.get(tid) or {}
        relative = Path(row.get("result_path") or "")
        result_path = (root / relative).resolve()
        if (
            counts[tid] != 1
            or not row
            or relative.is_absolute()
            or ".." in relative.parts
            or not result_path.is_relative_to(root)
            or result_path.name != "result.json"
            or result_path != Path(task.get("calculation_result_path") or "").resolve()
            or row.get("task_checksum") != task.get("task_checksum")
            or any(
                row.get(key) != task.get(key)
                for key in (
                    "task_id",
                    "task_key",
                    "batch_id",
                    "config_version",
                    "model_version",
                    "protocol_version",
                    "input_file_version",
                )
            )
            or row.get("input_structure_sha256") != task.get("input_structure_sha256")
        ):
            raise ValueError("phase_recovery_result_path_binding:" + tid)
        checked = verify_result(result_path, result_path.with_name("task.finished.json"), row)
        if not checked.get("valid"):
            raise ValueError("phase_recovery_receipt_invalid:" + tid)
        result, marker = checked["result"], checked["marker"]
        sidecar = json.loads(
            result_path.with_name("phase_identification.json").read_text(encoding="utf8")
        )
        old = (task.get("outputs") or {}).get("phase_identification") or {}
        if (
            sidecar.get("task_id") != tid
            or sidecar.get("task_checksum") != row["task_checksum"]
            or sidecar.get("result_checksum") != marker.get("result_checksum")
            or sidecar.get("phase_identification") != old
        ):
            raise ValueError("phase_recovery_original_evidence_mismatch:" + tid)
        output = deepcopy(result.get("outputs") or {})
        name = Path(str(output.get("structure_path") or "").replace("\\", "/")).name
        structure_path = (result_path.parent / name).resolve()
        saved_output = task.get("outputs") or {}
        if (
            not name
            or structure_path.parent != result_path.parent
            or structure_path != Path(saved_output.get("structure_path") or "").resolve()
            or file_checksum(structure_path) != output.get("structure_checksum")
            or output.get("structure_checksum") != saved_output.get("structure_checksum")
            or output.get("energy") != saved_output.get("energy")
        ):
            raise ValueError("phase_recovery_structure_binding:" + tid)
        if (
            output.get("single_point_completed") is True
            and (output.get("refresh_force_followup") or {}).get("relaxed") is False
            and output.get("structure_checksum") != row.get("input_structure_sha256")
        ):
            raise ValueError("phase_recovery_single_point_changed:" + tid)
        output.pop("phase_identification", None)
        output["structure_path"] = str(structure_path)
        identified = classify(
            {**result, "branch_id": row.get("branch_id"), "outputs": output},
            manager,
            phase_references=references,
            cache=None,
            scope_cache=scope_cache,
        )
        evidence = (identified.get("outputs") or {}).get("phase_identification") or {}
        proofs[tid] = {
            "evidence": evidence,
            "result_sha256": marker["result_checksum"],
            "task_checksum": row["task_checksum"],
            "files": {
                str(result_path): file_checksum(result_path),
                str(result_path.with_name("task.finished.json")): file_checksum(
                    result_path.with_name("task.finished.json")
                ),
                str(result_path.with_name("phase_identification.json")): file_checksum(
                    result_path.with_name("phase_identification.json")
                ),
                str(structure_path): file_checksum(structure_path),
            },
        }
    return proofs


def apply_verified_evidence(state, ledger, proofs, *, discard_ids, expected_model):
    """Apply independently verified proofs to the owned state and ledger copies."""
    if not isinstance(proofs, dict) or not proofs:
        raise ValueError("phase_recovery_empty_scope")
    discard_ids = set(discard_ids)
    if not discard_ids <= set(proofs):
        raise ValueError("phase_recovery_discard_scope")
    refresh = state.get("model_refresh") or {}
    if (
        state.get("active_model_version") != expected_model
        or refresh.get("new_model_version") != expected_model
        or refresh.get("status") != "waiting_results"
    ):
        raise ValueError("phase_recovery_model_changed")
    refresh_ids = refresh.get("task_ids") or []
    if len(set(refresh_ids)) != len(refresh_ids) or not set(proofs) <= set(refresh_ids):
        raise ValueError("phase_recovery_refresh_scope")
    tasks = state.get("tasks") or []
    records = state.get("phase_records") or []
    counts = Counter(t.get("task_id") for t in tasks)
    record_counts = Counter(r.get("source_task_id") for r in records)
    by_task = {t.get("task_id"): t for t in tasks}
    by_record = {r.get("source_task_id"): r for r in records}
    staged = []
    for tid, proof in proofs.items():
        task, record = by_task.get(tid) or {}, by_record.get(tid) or {}
        output = task.get("outputs") or {}
        old = output.get("phase_identification") or {}
        new = proof.get("evidence") or {}
        sid = task.get("structure_id")
        histories = ((ledger.get("structures") or {}).get(sid) or {}).get("stage_history") or {}
        history = [
            r
            for r in histories.get("relax_and_feature", [])
            if (r.get("metadata") or {}).get("task_id") == tid
        ]
        if (
            counts[tid] != 1
            or record_counts[tid] != 1
            or len(history) != 1
            or task.get("status") != "completed"
            or task.get("model_version") != expected_model
            or task.get("checks_passed", True) is not True
            or record.get("structure_id") != sid
            or record.get("source_version") != expected_model
            or record.get("status") != "pending_phase_identification"
            or old.get("status") != "unknown"
            or record.get("phase_identification") != old
            or (history[0].get("metadata") or {}).get("phase_identification") != old
            or old.get("structure_sha256") != output.get("structure_checksum")
            or new.get("structure_sha256") != old.get("structure_sha256")
            or new.get("portable_classification_scope") != old.get("portable_classification_scope")
            or not proof.get("result_sha256")
            or proof.get("task_checksum") != task.get("task_checksum")
        ):
            raise ValueError("phase_recovery_evidence_binding:" + tid)
        if tid in discard_ids:
            if (
                new.get("status") != "unknown"
                or new.get("reason") != "ValueError: unrecognized_phase:X"
            ):
                raise ValueError("phase_recovery_discard_not_unknown:" + tid)
        elif new.get("status") != "identified" or not new.get("phase"):
            raise ValueError("phase_recovery_not_identified:" + tid)
        staged.append((tid, task, record, history[0], old, new, proof))
    for tid, task, record, history, old, new, proof in staged:
        discarded = tid in discard_ids
        decision = "discard_unknown" if discarded else "identify"
        audit = {
            "decision": decision,
            "previous_evidence": deepcopy(old),
            "result_sha256": proof["result_sha256"],
            "task_checksum": proof["task_checksum"],
        }
        task["phase_evidence_recovery"] = audit
        output = task["outputs"]
        output["phase_identification"] = deepcopy(new)
        task["phase_identification_status"] = new["status"]
        record["phase_identification"] = deepcopy(new)
        record["phase_identification_status"] = new["status"]
        metadata = history.setdefault("metadata", {})
        metadata["phase_identification"] = deepcopy(new)
        if discarded:
            task.update(checks_passed=False, result_disposition="discarded_unknown_phase")
            record.update(checks_passed=False, status="excluded_unknown_phase")
            metadata["checks_passed"] = False
        else:
            phase = new["phase"]
            task.update(actual_phase=phase, observed_phase=phase)
            output.update(actual_phase=phase, observed_phase=phase)
            record.update(phase=phase, actual_phase=phase, observed_phase=phase, status="completed")
            metadata["actual_phase"] = phase
    state.setdefault("phase_evidence_recovery_history", []).append(
        {
            "model_version": expected_model,
            "task_ids": sorted(proofs),
            "discarded_task_ids": sorted(discard_ids),
        }
    )
    return {"identified": len(proofs) - len(discard_ids), "discarded": len(discard_ids)}
