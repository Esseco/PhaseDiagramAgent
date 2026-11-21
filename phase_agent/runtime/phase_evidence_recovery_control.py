"""Audited recovery of phase evidence for a frozen, returned refresh wave."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from filelock import FileLock


def load_frozen_phase_spec(archive_path, spec_path, spec_sha256):
    """Verify the unique frozen wave, including archive-only handoffs."""
    archive_path, spec_path = Path(archive_path).resolve(), Path(spec_path).resolve()
    with ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if n.endswith("/phase_wave.json")]
        if len(names) != 1:
            raise ValueError("phase_recovery_spec_not_in_frozen_archive")
        payload = archive.read(names[0])
    if hashlib.sha256(payload).hexdigest() != spec_sha256:
        raise ValueError("phase_recovery_frozen_source_changed")
    if spec_path != archive_path:
        external = spec_path.read_bytes()
        if hashlib.sha256(external).hexdigest() != spec_sha256:
            raise ValueError("phase_recovery_frozen_source_changed")
        if external != payload:
            raise ValueError("phase_recovery_spec_not_in_frozen_archive")
    return json.loads(payload)


def require_preserved_extrapolation_inputs(state, previous, updated):
    """Do not publish a latest-valid hull that silently loses an accepted prior frame."""
    refresh_ids = set((state.get("model_refresh") or {}).get("task_ids") or [])
    old = {
        (entry.get("structure_id"), entry.get("structure_sha256")): entry
        for entry in previous.get("entries") or []
    }
    new_entries = updated.get("entries") or []
    new_keys = {(entry.get("structure_id"), entry.get("structure_sha256")) for entry in new_entries}
    new_ids = {entry.get("record_id") for entry in new_entries}
    aliases = updated.get("equivalent_record_aliases") or {}
    for task in state.get("tasks") or []:
        if (
            task.get("task_id") not in refresh_ids
            or task.get("result_disposition") != "excluded_extrapolation_failure"
        ):
            continue
        key = (task.get("structure_id"), task.get("input_structure_sha256"))
        prior = old.get(key)
        if prior and key not in new_keys and aliases.get(prior.get("record_id")) not in new_ids:
            raise ValueError("phase_recovery_prior_energy_lost:" + str(task.get("task_id")))


def recover(
    control,
    *,
    expected_model,
    expected_hull,
    epoch_root,
    spec_path,
    archive_path,
    spec_sha256,
    archive_sha256,
    task_ids,
    discard_task_ids,
    user_instruction,
    hull_source_policy,
):
    """Verify frozen inputs, repair bound evidence, and atomically persist a new hull."""
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    from phase_agent.runtime.phase_evidence_recovery import (
        apply_verified_evidence,
        prepare_bound_proofs,
    )
    from phase_agent.tools.remote.integrity import file_checksum
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError("phase_recovery_explicit_instruction_required")
    if hull_source_policy not in {"current_model_only_v1", "latest_valid_per_structure_v1"}:
        raise ValueError("phase_recovery_hull_policy_requires_explicit_instruction")
    ids = list(task_ids)
    dropped = list(discard_task_ids)
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(dropped) != len(set(dropped))
        or not set(dropped) <= set(ids)
    ):
        raise ValueError("phase_recovery_invalid_scope")
    state_path = Path(control.state_path).resolve()
    project_root = state_path.parent.parent.resolve()
    root, spec_path, archive_path = map(
        lambda p: Path(p).resolve(), (epoch_root, spec_path, archive_path)
    )
    if not all(p.is_relative_to(project_root) for p in (root, spec_path, archive_path)):
        raise ValueError("phase_recovery_path_outside_project")
    if file_checksum(archive_path) != archive_sha256:
        raise ValueError("phase_recovery_frozen_source_changed")
    spec = load_frozen_phase_spec(archive_path, spec_path, spec_sha256)
    with (
        control.chat_handler.lock,
        FileLock(str(state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        state = read_json(state_path, {}) or {}
        from phase_agent.tools.state.workflow_control import read_control

        prior = (read_control(state_path) or {}).get("evidence") or {}
        matching_receipt = (
            prior.get("operation") == "recover_phase_evidence"
            and prior.get("model_version") == expected_model
            and prior.get("previous_hull") == expected_hull
            and prior.get("spec_sha256") == spec_sha256
            and prior.get("archive_sha256") == archive_sha256
            and prior.get("hull_source_policy") == hull_source_policy
            and prior.get("task_ids") == sorted(ids)
            and prior.get("discarded_task_ids") == sorted(dropped)
        )
        if matching_receipt and state.get("active_model_version") == expected_model:
            by_id = {t.get("task_id"): t for t in state.get("tasks") or []}
            if all(
                sum(t.get("task_id") == tid for t in state.get("tasks") or []) == 1
                and (by_id.get(tid, {}).get("phase_evidence_recovery") or {}).get("decision")
                == ("discard_unknown" if tid in dropped else "identify")
                for tid in ids
            ):
                return {
                    "status": "already_registered",
                    "identified": len(ids) - len(dropped),
                    "discarded": len(dropped),
                    "hull_version": ((state.get("phase_diagrams") or {}).get("mlip") or {}).get(
                        "version"
                    ),
                    "submitted": False,
                }
        if (
            state.get("active_model_version") != expected_model
            or (state.get("phase_diagrams", {}).get("mlip") or {}).get("version") != expected_hull
        ):
            raise ValueError("phase_recovery_snapshot_changed")
        manager = control.chat_handler.workflow_kwargs.get("manager")
        ledger_path = Path(getattr(manager, "_ledger_path", "") or "").resolve()
        if (
            not ledger_path.is_relative_to(state_path.parent / "ledgers")
            or not ledger_path.is_file()
        ):
            raise ValueError("phase_recovery_ledger_unavailable")
        state_hash, ledger_hash = file_checksum(state_path), file_checksum(ledger_path)
        ledger = read_json(ledger_path, {}) or {}
        proofs = prepare_bound_proofs(state, spec, root, ids)
        summary = apply_verified_evidence(
            state, ledger, proofs, discard_ids=dropped, expected_model=expected_model
        )
        state["model_refresh"]["hull_source_policy"] = hull_source_policy
        state["phase_evidence_recovery_history"][-1]["hull_source_policy"] = hull_source_policy
        records = state.get("phase_records") or []
        diagrams = update_phase_diagram(
            records,
            active_model_version=expected_model,
            output_state=state,
            latest_per_structure=True,
            latest_only=True,
        )
        diagram = (diagrams.get("diagrams") or {}).get("mlip") or {}
        if (
            diagram.get("status") != "completed"
            or diagram.get("model_version") != expected_model
            or (
                hull_source_policy == "current_model_only_v1"
                and any(
                    (e.get("model_version") or e.get("source_version")) != expected_model
                    for e in diagram.get("entries") or []
                )
            )
        ):
            raise ValueError("phase_recovery_hull_invalid")
        if hull_source_policy == "latest_valid_per_structure_v1":
            previous = (state.get("phase_diagrams") or {}).get("mlip") or {}
            require_preserved_extrapolation_inputs(state, previous, diagram)
        state.setdefault("phase_diagrams", {})["mlip"] = diagram
        by_model = update_phase_diagram(
            records,
            active_model_version=expected_model,
            output_state=state,
            model_versions_only=[expected_model],
        )
        active = (by_model.get("mlip_by_version") or {}).get(expected_model) or {}
        if active.get("status") != "completed":
            raise ValueError("phase_recovery_model_hull_invalid")
        state.setdefault("phase_diagrams_by_model", {})[expected_model] = active
        state["phase_diagram_input_checksum"] = hashlib.sha256(
            json.dumps(
                sorted(
                    records,
                    key=lambda row: (
                        str(row.get("record_id") or ""),
                        str(row.get("structure_id") or ""),
                    ),
                ),
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        directory = state.get("analysis_output_directory")
        state["analysis_output_directory"] = None
        state, waiting = advance_refresh(state, _owned=True)
        state["analysis_output_directory"] = directory
        if waiting or (state.get("model_refresh") or {}).get("status") != "completed":
            raise ValueError("phase_recovery_refresh_blocked:" + str(waiting))
        if file_checksum(archive_path) != archive_sha256:
            raise ValueError("phase_recovery_frozen_source_changed")
        load_frozen_phase_spec(archive_path, spec_path, spec_sha256)
        for proof in proofs.values():
            for name, digest in proof.get("files", {}).items():
                if file_checksum(name) != digest:
                    raise ValueError("phase_recovery_evidence_changed:" + name)
        commit = commit_result_recovery(
            state_path,
            state,
            {str(ledger_path): ledger},
            expected_hashes={str(state_path): state_hash, str(ledger_path): ledger_hash},
            evidence={
                "operation": "recover_phase_evidence",
                "model_version": expected_model,
                "previous_hull": expected_hull,
                "task_ids": sorted(ids),
                "discarded_task_ids": sorted(dropped),
                "spec_sha256": spec_sha256,
                "archive_sha256": archive_sha256,
                "user_instruction": user_instruction.strip(),
                "hull_source_policy": hull_source_policy,
            },
        )
        control._codex_context_cache = None
        return {
            "status": "completed",
            **summary,
            "hull_version": diagram.get("version"),
            "entry_count": len(diagram.get("entries") or []),
            "mixed_model_versions": diagram.get("mixed_model_versions"),
            "recovery_revision": commit["revision"],
            "submitted": False,
        }
