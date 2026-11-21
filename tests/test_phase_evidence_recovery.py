from copy import deepcopy

import pytest

from phase_agent.runtime.phase_evidence_recovery import apply_verified_evidence


def fixture():
    old = {
        "status": "unknown",
        "structure_sha256": "hash",
        "portable_classification_scope": "scope",
    }
    task = {
        "task_id": "t",
        "task_checksum": "receipt",
        "status": "completed",
        "model_version": "m",
        "structure_id": "s",
        "outputs": {"structure_checksum": "hash", "phase_identification": old},
    }
    record = {
        "source_task_id": "t",
        "structure_id": "s",
        "source_version": "m",
        "status": "pending_phase_identification",
        "phase_identification": deepcopy(old),
    }
    ledger = {
        "structures": {
            "s": {
                "stage_history": {
                    "relax_and_feature": [
                        {"metadata": {"task_id": "t", "phase_identification": deepcopy(old)}}
                    ]
                }
            }
        }
    }
    state = {
        "active_model_version": "m",
        "model_refresh": {"new_model_version": "m", "status": "waiting_results", "task_ids": ["t"]},
        "tasks": [task],
        "phase_records": [record],
    }
    evidence = {
        "status": "identified",
        "phase": "P3",
        "structure_sha256": "hash",
        "portable_classification_scope": "scope",
    }
    proof = {"evidence": evidence, "result_sha256": "result", "task_checksum": "receipt"}
    return state, ledger, {"t": proof}


def test_identification_updates_bound_records_and_preserves_prior_evidence():
    state, ledger, proofs = fixture()
    assert apply_verified_evidence(state, ledger, proofs, discard_ids=[], expected_model="m") == {
        "identified": 1,
        "discarded": 0,
    }
    assert state["tasks"][0]["outputs"]["actual_phase"] == "P3"
    assert state["phase_records"][0]["phase"] == "P3"
    assert state["tasks"][0]["phase_evidence_recovery"]["previous_evidence"]["status"] == "unknown"
    assert (
        ledger["structures"]["s"]["stage_history"]["relax_and_feature"][0]["metadata"][
            "actual_phase"
        ]
        == "P3"
    )
    with pytest.raises(ValueError):
        apply_verified_evidence(state, ledger, proofs, discard_ids=[], expected_model="m")


def test_discard_requires_actual_unknown_and_records_decision():
    state, ledger, proofs = fixture()
    with pytest.raises(ValueError):
        apply_verified_evidence(state, ledger, proofs, discard_ids=["t"], expected_model="m")
    proofs["t"]["evidence"].update(status="unknown", reason="ValueError: unrecognized_phase:X")
    apply_verified_evidence(state, ledger, proofs, discard_ids=["t"], expected_model="m")
    assert state["tasks"][0]["result_disposition"] == "discarded_unknown_phase"
    assert state["phase_records"][0]["status"] == "excluded_unknown_phase"


@pytest.mark.parametrize("field,value", [("task_checksum", "bad"), ("result_sha256", "")])
def test_wrong_receipt_fails_without_mutating_state(field, value):
    state, ledger, proofs = fixture()
    before = deepcopy((state, ledger))
    proofs["t"][field] = value
    with pytest.raises(ValueError):
        apply_verified_evidence(state, ledger, proofs, discard_ids=[], expected_model="m")
    assert (state, ledger) == before


def test_prepare_requires_unchanged_receipts_and_exact_structure(tmp_path):
    import hashlib
    import json
    from phase_agent.runtime.phase_evidence_recovery import prepare_bound_proofs

    folder = tmp_path / "calculations" / "results" / "one"
    folder.mkdir(parents=True)
    frame = folder / "initial.vasp"
    frame.write_bytes(b"frame")
    digest = hashlib.sha256(b"frame").hexdigest()
    old = {
        "status": "unknown",
        "structure_sha256": digest,
        "portable_classification_scope": "scope",
    }
    row = {
        "task_id": "t",
        "task_key": "key",
        "task_checksum": "receipt",
        "model_version": "m",
        "input_structure_sha256": digest,
        "result_path": "calculations/results/one/result.json",
    }
    result = {
        "task_id": "t",
        "task_key": "key",
        "model_version": "m",
        "status": "completed",
        "outputs": {
            "structure_path": "initial.vasp",
            "structure_checksum": digest,
            "energy": -1,
            "single_point_completed": True,
            "refresh_force_followup": {"relaxed": False},
        },
    }
    result_path = folder / "result.json"
    result_path.write_text(json.dumps(result))
    result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    marker = {
        "task_id": "t",
        "task_key": "key",
        "model_version": "m",
        "status": "completed",
        "task_checksum": "receipt",
        "result_checksum": result_digest,
    }
    (folder / "task.finished.json").write_text(json.dumps(marker))
    (folder / "phase_identification.json").write_text(
        json.dumps(
            {
                "task_id": "t",
                "task_checksum": "receipt",
                "result_checksum": result_digest,
                "phase_identification": old,
            }
        )
    )
    task = {
        **row,
        "calculation_result_path": str(result_path),
        "outputs": {
            "structure_path": str(frame),
            "structure_checksum": digest,
            "energy": -1,
            "phase_identification": old,
        },
    }
    spec = {"tasks": [row], "manager": {"boundary": {}}, "references": {}}

    def classify(result, manager, **kwargs):
        return {
            "outputs": {
                "phase_identification": {
                    "status": "identified",
                    "phase": "P3",
                    "structure_sha256": digest,
                    "portable_classification_scope": "scope",
                }
            }
        }

    proof = prepare_bound_proofs({"tasks": [task]}, spec, tmp_path, ["t"], classify=classify)
    assert proof["t"]["result_sha256"] == result_digest
    frame.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="structure_binding"):
        prepare_bound_proofs({"tasks": [task]}, spec, tmp_path, ["t"], classify=classify)


def test_control_checks_frozen_archive_and_commits_bound_evidence(tmp_path, monkeypatch):
    import hashlib
    import json
    import threading
    from types import SimpleNamespace
    from zipfile import ZipFile
    from phase_agent.runtime.phase_evidence_recovery_control import recover
    import phase_agent.runtime.phase_evidence_recovery as recovery
    import phase_agent.analysis.phase.update_phase_diagram as hull_module
    import phase_agent.tools.workflows.model_refresh_gate as gate
    import phase_agent.persistence.result_recovery_commit as commit_module

    state, ledger, proofs = fixture()
    state["phase_diagrams"] = {"mlip": {"version": "h0"}}
    state_dir = tmp_path / "workflow_state"
    ledger_dir = state_dir / "ledgers"
    ledger_dir.mkdir(parents=True)
    state_path, ledger_path = state_dir / "state.json", ledger_dir / "phase_data.json"
    state_path.write_text(json.dumps(state))
    ledger_path.write_text(json.dumps(ledger))
    epoch = tmp_path / "epoch"
    epoch.mkdir()
    spec_path, archive = epoch / "phase_wave.json", epoch / "inputs.zip"
    spec_path.write_text('{"tasks": []}')
    with ZipFile(archive, "w") as z:
        z.writestr("calculations/phase_wave.json", spec_path.read_bytes())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(recovery, "prepare_bound_proofs", lambda *a, **k: deepcopy(proofs))

    def fake_hull(*args, **kwargs):
        if kwargs.get("latest_only"):
            return {
                "diagrams": {
                    "mlip": {
                        "status": "completed",
                        "model_version": "m",
                        "version": "h1",
                        "entries": [],
                    }
                }
            }
        return {"mlip_by_version": {"m": {"status": "completed"}}}

    monkeypatch.setattr(hull_module, "update_phase_diagram", fake_hull)

    def fake_advance(state, **kwargs):
        state["model_refresh"]["status"] = "completed"
        return state, None

    monkeypatch.setattr(gate, "advance_refresh", fake_advance)
    commits = []
    original_commit = commit_module.commit_result_recovery

    def checked_commit(path, desired, files, **kwargs):
        commits.append((desired, files, kwargs))
        return original_commit(path, desired, files, **kwargs)

    monkeypatch.setattr(commit_module, "commit_result_recovery", checked_commit)
    handler = SimpleNamespace(
        lock=threading.RLock(),
        workflow_kwargs={"manager": SimpleNamespace(_ledger_path=ledger_path)},
    )
    control = SimpleNamespace(
        state_path=state_path, chat_handler=handler, _codex_context_cache=None
    )
    args = dict(
        expected_model="m",
        expected_hull="h0",
        epoch_root=epoch,
        spec_path=spec_path,
        archive_path=archive,
        spec_sha256=digest(spec_path),
        archive_sha256=digest(archive),
        task_ids=["t"],
        discard_task_ids=[],
        user_instruction="identify",
        hull_source_policy="current_model_only_v1",
    )
    out = recover(control, **args)
    assert out["status"] == "completed" and out["identified"] == 1
    assert commits[0][0]["phase_records"][0]["phase"] == "P3"
    assert (
        commits[0][1][str(ledger_path)]["structures"]["s"]["stage_history"]["relax_and_feature"][0][
            "metadata"
        ]["actual_phase"]
        == "P3"
    )
    assert commits[0][2]["evidence"]["task_ids"] == ["t"]
    assert recover(control, **args)["status"] == "already_registered"
    assert len(commits) == 1
    assert json.loads(state_path.read_text(encoding="utf8"))["phase_records"][0]["phase"] == "P3"
    assert (
        json.loads(ledger_path.read_text(encoding="utf8"))["structures"]["s"]["stage_history"][
            "relax_and_feature"
        ][0]["metadata"]["actual_phase"]
        == "P3"
    )
    spec_path.write_text("changed")
    with pytest.raises(ValueError, match="frozen_source_changed"):
        recover(control, **args)
    assert len(commits) == 1
