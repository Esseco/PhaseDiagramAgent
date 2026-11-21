"""Project-bound refresh fallback and frozen input compatibility."""

import hashlib
import json
from zipfile import ZipFile

import pytest

from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
from phase_agent.runtime.phase_evidence_recovery_control import load_frozen_phase_spec


def test_latest_valid_policy_keeps_prior_extrapolation_frame_only_when_authorized():
    def record(rid, sid, sha, model, na, energy):
        return {
            "record_id": rid,
            "structure_id": sid,
            "structure_sha256": sha,
            "model_version": model,
            "source_version": model,
            "energy_method": "mlip",
            "energy_unit": "eV",
            "energy": energy,
            "status": "completed",
            "composition": {"Na": na, "Cr": 1, "O": 2},
            "phase": "O3",
        }

    task = {
        "task_id": "failed",
        "task_checksum": "receipt",
        "status": "completed",
        "stage": "relax_and_feature",
        "checks_passed": False,
        "result_disposition": "excluded_extrapolation_failure",
        "model_version": "new",
        "structure_id": "s",
        "input_structure_sha256": "input",
        "parameters": {"model_refresh_operation": "relax"},
        "outputs": {"extrapolation_monitor": {"extrapolation_failure": True}},
    }
    records = [
        record("zero", "z", "zero", "new", 0, -5),
        record("one", "o", "one", "new", 1, -7),
        record("old", "s", "input", "old", 0.5, -10),
    ]
    state = {
        "tasks": [task],
        "upload_layout": {"model_rounds": {"old": 1, "new": 2}},
        "model_refresh": {"hull_source_policy": "latest_valid_per_structure_v1"},
    }
    diagram = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert {r["record_id"] for r in diagram["entries"]} == {"zero", "one", "old"}
    state["mlip_energy_policy"] = "latest_valid_per_structure_v1"
    state["model_refresh"].pop("hull_source_policy")
    diagram = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )["diagrams"]["mlip"]
    assert {r["record_id"] for r in diagram["entries"]} == {"zero", "one", "old"}
    state["model_refresh"]["hull_source_policy"] = "current_model_only_v1"
    diagram = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )["diagrams"]["mlip"]
    assert {r["record_id"] for r in diagram["entries"]} == {"zero", "one"}


def test_spec_can_be_verified_inside_frozen_archive(tmp_path):
    payload = json.dumps({"tasks": [{"task_id": "t"}]}).encode()
    archive = tmp_path / "inputs.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("calculations/a/inputs/zz-phase-analysis-b/phase_wave.json", payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert load_frozen_phase_spec(archive, archive, digest) == {"tasks": [{"task_id": "t"}]}
    with pytest.raises(ValueError, match="phase_recovery_frozen_source_changed"):
        load_frozen_phase_spec(archive, archive, "0" * 64)
    path = tmp_path / "phase_wave.json"
    path.write_bytes(payload)
    assert load_frozen_phase_spec(archive, path, digest) == {"tasks": [{"task_id": "t"}]}
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_frozen_phase_spec(archive, path, digest)


def test_recovery_transaction_accepts_mixed_latest_policy_and_archive_spec(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import phase_agent.runtime.phase_evidence_recovery as recovery
    import phase_agent.analysis.phase.update_phase_diagram as hull_module
    import phase_agent.tools.workflows.model_refresh_gate as gate
    import phase_agent.persistence.result_recovery_commit as commit
    from phase_agent.runtime.phase_evidence_recovery_control import recover

    state_dir = tmp_path / "workflow_state"
    ledger_dir = state_dir / "ledgers"
    ledger_dir.mkdir(parents=True)
    state_path = state_dir / "state.json"
    ledger_path = ledger_dir / "phase_data.json"
    state_path.write_text(
        json.dumps(
            {
                "active_model_version": "m",
                "model_refresh": {"status": "waiting_results", "new_model_version": "m"},
                "phase_diagrams": {"mlip": {"version": "h0"}},
                "phase_records": [],
            }
        )
    )
    ledger_path.write_text("{}")
    epoch = tmp_path / "epoch4"
    epoch.mkdir()
    archive = epoch / "inputs.zip"
    payload = b'{"tasks":[]}'
    with ZipFile(archive, "w") as z:
        z.writestr("calculations/a/inputs/b/phase_wave.json", payload)
    monkeypatch.setattr(recovery, "prepare_bound_proofs", lambda *a, **k: {"t": {"files": {}}})
    monkeypatch.setattr(
        recovery,
        "apply_verified_evidence",
        lambda state, ledger, proofs, **kw: (
            state.setdefault("phase_evidence_recovery_history", [{}])
            and {"identified": 1, "discarded": 0}
        ),
    )

    def hull(*args, **kwargs):
        if kwargs.get("latest_only"):
            return {
                "diagrams": {
                    "mlip": {
                        "status": "completed",
                        "model_version": "m",
                        "version": "h1",
                        "entries": [{"source_version": "old"}],
                    }
                }
            }
        return {"mlip_by_version": {"m": {"status": "completed"}}}

    monkeypatch.setattr(hull_module, "update_phase_diagram", hull)

    def advance(state, **kwargs):
        state["model_refresh"]["status"] = "completed"
        return state, None

    monkeypatch.setattr(gate, "advance_refresh", advance)
    captured = []
    monkeypatch.setattr(
        commit,
        "commit_result_recovery",
        lambda path, desired, files, **kw: captured.append((desired, kw)) or {"revision": "r"},
    )
    control = SimpleNamespace(
        state_path=state_path,
        chat_handler=SimpleNamespace(
            lock=threading.RLock(),
            workflow_kwargs={"manager": SimpleNamespace(_ledger_path=ledger_path)},
        ),
        _codex_context_cache=None,
    )
    out = recover(
        control,
        expected_model="m",
        expected_hull="h0",
        epoch_root=epoch,
        spec_path=archive,
        archive_path=archive,
        spec_sha256=hashlib.sha256(payload).hexdigest(),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        task_ids=["t"],
        discard_task_ids=[],
        user_instruction="保留旧有效能量",
        hull_source_policy="latest_valid_per_structure_v1",
    )
    assert out["status"] == "completed"
    assert captured[0][0]["model_refresh"]["hull_source_policy"] == "latest_valid_per_structure_v1"
    assert captured[0][1]["evidence"]["hull_source_policy"] == "latest_valid_per_structure_v1"


def test_prior_energy_policy_does_not_revive_explicitly_discarded_unknown():
    from phase_agent.analysis.phase.outside_refresh_hull import rejected_refresh_inputs

    task = {
        "task_id": "t",
        "task_checksum": "checksum",
        "status": "completed",
        "stage": "relax_and_feature",
        "checks_passed": False,
        "result_disposition": "discarded_unknown_phase",
        "model_version": "new",
        "structure_id": "s",
        "input_structure_sha256": "hash",
        "parameters": {"model_refresh_operation": "relax"},
        "outputs": {
            "phase_identification": {
                "status": "unknown",
                "reason": "ValueError: unrecognized_phase:X",
            }
        },
        "phase_evidence_recovery": {"decision": "discard_unknown"},
    }
    record = {
        "source_task_id": "t",
        "structure_id": "s",
        "model_version": "new",
        "checks_passed": False,
        "status": "excluded_unknown_phase",
    }
    assert rejected_refresh_inputs([record], [task], {"new": 2}, preserve_extrapolation=True) == {
        ("s", "hash"): 2
    }


def test_transaction_rejects_losing_prior_extrapolation_input():
    from phase_agent.runtime.phase_evidence_recovery_control import (
        require_preserved_extrapolation_inputs,
    )

    task = {
        "task_id": "t",
        "structure_id": "s",
        "input_structure_sha256": "h",
        "result_disposition": "excluded_extrapolation_failure",
    }
    state = {"tasks": [task], "model_refresh": {"task_ids": ["t"]}}
    previous = {"entries": [{"record_id": "old", "structure_id": "s", "structure_sha256": "h"}]}
    with pytest.raises(ValueError, match="phase_recovery_prior_energy_lost"):
        require_preserved_extrapolation_inputs(state, previous, {"entries": []})
    require_preserved_extrapolation_inputs(
        state,
        previous,
        {"entries": [{"record_id": "new", "structure_id": "s", "structure_sha256": "h"}]},
    )
    require_preserved_extrapolation_inputs(
        state,
        previous,
        {
            "entries": [{"record_id": "equivalent"}],
            "equivalent_record_aliases": {"old": "equivalent"},
        },
    )


def test_phase_evidence_reuses_scope_cache_within_frozen_wave(tmp_path):
    from phase_agent.runtime.phase_evidence_recovery import prepare_bound_proofs

    rows = []
    tasks = []
    for tid in ("a", "b"):
        folder = tmp_path / tid
        folder.mkdir()
        frame = folder / "initial.vasp"
        frame.write_bytes(tid.encode())
        digest = hashlib.sha256(frame.read_bytes()).hexdigest()
        row = {
            "task_id": tid,
            "task_key": tid,
            "task_checksum": "receipt-" + tid,
            "model_version": "m",
            "input_structure_sha256": digest,
            "result_path": tid + "/result.json",
        }
        outputs = {
            "structure_path": "initial.vasp",
            "structure_checksum": digest,
            "energy": -1,
            "single_point_completed": True,
            "refresh_force_followup": {"relaxed": False},
        }
        result = {
            "task_id": tid,
            "task_key": tid,
            "model_version": "m",
            "status": "completed",
            "outputs": outputs,
        }
        result_path = folder / "result.json"
        result_path.write_text(json.dumps(result))
        result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        marker = {
            "task_id": tid,
            "task_key": tid,
            "model_version": "m",
            "status": "completed",
            "task_checksum": row["task_checksum"],
            "result_checksum": result_hash,
        }
        (folder / "task.finished.json").write_text(json.dumps(marker))
        old = {
            "status": "unknown",
            "structure_sha256": digest,
            "portable_classification_scope": "scope",
        }
        (folder / "phase_identification.json").write_text(
            json.dumps(
                {
                    "task_id": tid,
                    "task_checksum": row["task_checksum"],
                    "result_checksum": result_hash,
                    "phase_identification": old,
                }
            )
        )
        tasks.append(
            {
                **row,
                "calculation_result_path": str(result_path),
                "outputs": {**outputs, "structure_path": str(frame), "phase_identification": old},
            }
        )
        rows.append(row)
    caches = []

    def classify(result, manager, **kwargs):
        caches.append(kwargs["scope_cache"])
        return {
            "outputs": {
                "phase_identification": {
                    "status": "identified",
                    "phase": "O3",
                    "structure_sha256": result["outputs"]["structure_checksum"],
                    "portable_classification_scope": "scope",
                }
            }
        }

    proofs = prepare_bound_proofs(
        {"tasks": tasks},
        {"tasks": rows, "manager": {"boundary": {}}, "references": {}},
        tmp_path,
        ["a", "b"],
        classify=classify,
    )
    assert set(proofs) == {"a", "b"} and caches[0] is caches[1]
