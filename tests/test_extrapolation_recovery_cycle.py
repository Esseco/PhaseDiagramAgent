import hashlib

import pytest

from phase_agent.evidence.extrapolation_recovery import quarantine_refresh, recovery_label_facts
from phase_agent.tools.workflows.model_refresh_state import refresh_active


def test_quarantine_preserves_old_hull_and_is_idempotent(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    old = {"status": "completed", "version": "old-hull", "model_version": "old", "entries": []}
    new = {"status": "completed", "version": "bad-hull", "model_version": "bad", "entries": []}
    frame = tmp_path / "frame.vasp"
    frame.write_bytes(b"frame")
    checksum = hashlib.sha256(frame.read_bytes()).hexdigest()
    state = {
        "active_model_version": "bad",
        "phase_diagrams": {"mlip": new},
        "model_refresh": {
            "status": "waiting_results",
            "old_model_version": "old",
            "new_model_version": "bad",
            "old_diagram": old,
            "task_ids": ["r1"],
        },
        "tasks": [{"task_id": "r1", "model_version": "bad"}],
        "direct_sampling_results": {
            "epoch_refresh": {
                "model_version": "bad",
                "source_digest": "digest",
                "selected_count": 1,
                "selected": [
                    {
                        "candidate_id": "c",
                        "source_task_id": "r1",
                        "structure_path": str(frame),
                        "structure_sha256": checksum,
                    }
                ],
            }
        },
    }
    updated, receipt = quarantine_refresh(state, "epoch_refresh")
    assert receipt["status"] == "quarantined" and updated["phase_diagrams"]["mlip"] == old
    assert state["phase_diagrams"]["mlip"] == new
    assert updated["extrapolation_recovery_cycle"]["invalidated_task_ids"] == ["r1"]
    assert not refresh_active(updated)
    assert quarantine_refresh(updated, "epoch_refresh")[1]["status"] == "already_quarantined"


def test_only_exact_completed_dft_labels_unlock_retraining(tmp_path, monkeypatch):
    from phase_agent.science.dft import spin_acceptance
    from phase_agent.policies.workflow_policy import action_errors

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda record: True)
    frame = tmp_path / "input.vasp"
    frame.write_bytes(b"frame")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    cycle = {"status": "awaiting_reviewed_dft_labels", "selected": [{"structure_sha256": digest}]}
    state = {"extrapolation_recovery_cycle": cycle, "tasks": [], "dft_dataset_records": []}
    assert recovery_label_facts(state)["status"] == "waiting"
    assert "extrapolation_recovery_dft_labels_required" in action_errors(
        {"tool": "update_mlip"}, state, {}
    )
    state["tasks"] = [
        {
            "task_id": "d",
            "stage": "dft_single_point",
            "status": "completed",
            "structure_path": str(frame),
        }
    ]
    state["dft_dataset_records"] = [
        {
            "task_id": "d",
            "status": "completed",
            "converged": True,
            "checks_passed": True,
            "training_ready": True,
            "final_frame_valid": True,
        }
    ]
    assert recovery_label_facts(state)["status"] == "complete"
    from phase_agent.evidence.extrapolation_recovery import recovery_model_matches_labels
    cycle["invalid_model_version"] = "bad"
    state["remote_finetune_jobs"] = {"j": {"status": "results_received", "original_model_version": "bad",
        "training_handoff": {"candidate_model_version": "new"}, "report": {"dataset_scope": {"data_ids": ["other:frame:0"]}}}}
    assert not recovery_model_matches_labels(state, "new")
    state["remote_finetune_jobs"]["j"]["report"]["dataset_scope"]["data_ids"] = ["d:frame:0"]
    assert recovery_model_matches_labels(state, "new")
    frame.write_bytes(b"different")
    assert recovery_label_facts(state)["status"] == "waiting"


def test_unknown_previous_hull_cannot_be_quarantined(monkeypatch):
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    with pytest.raises(ValueError, match="previous_hull_invalid"):
        quarantine_refresh(
            {
                "active_model_version": "m",
                "model_refresh": {"new_model_version": "m", "task_ids": ["t"]},
                "direct_sampling_results": {"epoch_refresh": {"model_version": "m"}},
            },
            "epoch_refresh",
        )


def test_replacement_model_refresh_starts_from_previous_valid_hull(monkeypatch):
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh

    baseline = {"status": "completed", "version": "old-hull", "model_version": "old"}
    state = {
        "phase_diagrams": {"mlip": baseline},
        "extrapolation_recovery_cycle": {
            "status": "awaiting_reviewed_dft_labels",
            "invalid_model_version": "bad",
            "baseline_model_version": "old",
            "baseline_hull_version": "old-hull",
            "baseline_diagram": baseline,
        },
    }
    monkeypatch.setattr(
        extrapolation_recovery, "recovery_label_facts", lambda state: {"status": "waiting"}
    )
    with pytest.raises(ValueError, match="labels_incomplete"):
        require_structure_refresh(state, "bad", "new")
    monkeypatch.setattr(
        extrapolation_recovery, "recovery_label_facts", lambda state: {"status": "complete"}
    )
    require_structure_refresh(state, "bad", "new")
    assert state["model_refresh"]["old_model_version"] == "old"
    assert state["model_refresh"]["old_diagram"] == baseline
    assert state["extrapolation_recovery_cycle"]["training_parent_model_version"] == "bad"


def test_complete_labels_trigger_strategic_training_proposal(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    monkeypatch.setattr(extrapolation_round, "required_direct_round", lambda state: None)
    monkeypatch.setattr(extrapolation_recovery, "recovery_context", lambda state: {
        "status": "awaiting_reviewed_dft_labels", "labels": {"status": "complete"}})
    state = {"extrapolation_recovery_cycle": {"status": "awaiting_reviewed_dft_labels", "invalid_model_version": "bad"}}
    action = {"tool": "update_mlip", "operation": "prepare_mlip_training", "parameters": {
        "action": "RETRAIN_MLIP", "prepare_inputs_only": True}, "task_key": "train", "budget": 0.0}
    result = select_initial_proposal(current=state, mode="interactive", config={"agent": {"allowed_tools": ["update_mlip"]}},
        context={}, decision_state={}, registry={"update_mlip": {"handler": lambda **kw: None}},
        agent_client=lambda payload: None, invocation_id="test", propose=lambda *args, **kwargs: action,
        revise=None, prepare_debug=None, is_verified_second=None, mc_block=None, model_failed=None,
        record_id_factory=lambda *args: "record")
    assert result["proposal"]["raw_action"]["operation"] == "prepare_mlip_training"


def test_publication_quarantine_restores_previous_bundle_and_preserves_history(
    tmp_path, monkeypatch
):
    import json
    from phase_agent.analysis.feedback.versioned_products import publish_bundle, validate_bundle
    from phase_agent.tools.workflows import direct_extrapolation_sampling
    from phase_agent.tools.workflows.quarantine_refresh_publication import (
        quarantine_refresh_publication,
    )

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    project = tmp_path / "project"
    root = project / "analysis_outputs"
    state_path = project / "workflow_state" / "state.json"
    state_path.parent.mkdir(parents=True)

    def build(directory):
        (directory / "report.md").write_text("evidence", encoding="utf8")

    publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch0",
        model="old",
        identity={"hull_version": "old-hull"},
        required=["report.md"],
        build=build,
    )
    bad = publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch1",
        model="bad",
        identity={"hull_version": "bad-hull"},
        required=["report.md"],
        build=build,
    )
    frame = project / "frame.vasp"
    frame.write_bytes(b"frame")
    checksum = hashlib.sha256(frame.read_bytes()).hexdigest()
    old = {"status": "completed", "version": "old-hull", "model_version": "old", "entries": []}
    state = {
        "analysis_output_directory": str(root),
        "active_model_version": "bad",
        "upload_layout": {"model_rounds": {"old": 1, "bad": 2}},
        "phase_diagrams": {
            "mlip": {"status": "completed", "version": "bad-hull", "model_version": "bad"}
        },
        "model_refresh": {
            "status": "waiting_results",
            "old_model_version": "old",
            "new_model_version": "bad",
            "old_diagram": old,
            "task_ids": ["r1"],
        },
        "tasks": [{"task_id": "r1", "model_version": "bad"}],
        "direct_sampling_results": {
            "epoch_refresh": {
                "model_version": "bad",
                "source_digest": "digest",
                "selected_count": 1,
                "selected": [
                    {
                        "candidate_id": "c",
                        "source_task_id": "r1",
                        "structure_path": str(frame),
                        "structure_sha256": checksum,
                    }
                ],
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf8")
    result = quarantine_refresh_publication(state, state_path, "epoch_refresh")
    assert result["status"] == "quarantined"
    assert (
        validate_bundle(root / "latest" / "phase_diagram")["identity"]["hull_version"] == "old-hull"
    )
    assert validate_bundle(bad["directory"])["identity"]["hull_version"] == "bad-hull"
    invalid_status = json.loads(
        (root / "epoch1" / "phase_diagram" / "latest_status.json").read_text(encoding="utf8")
    )
    assert invalid_status["status"] == "invalidated"
    assert "bad-hull" in (root / "epoch1" / "phase_diagram" / "INVALIDATED.md").read_text(
        encoding="utf8"
    )
    stored = json.loads(state_path.read_text(encoding="utf8"))
    assert stored["extrapolation_recovery_cycle"]["status"] == "awaiting_reviewed_dft_labels"
    assert (
        quarantine_refresh_publication(stored, state_path, "epoch_refresh")["status"]
        == "already_quarantined"
    )
    from phase_agent.analysis.feedback import failed_epoch_archive
    from phase_agent.analysis.feedback.failed_epoch_archive import archive_failed_epoch

    def fail_copy(*args):
        raise OSError("synthetic copy failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(failed_epoch_archive.shutil, "copyfile", fail_copy)
        with pytest.raises(OSError, match="synthetic copy failure"):
            archive_failed_epoch(stored, state_path)
    assert not (root / "epoch1-extrapolation_failed").exists()
    assert not list(root.glob(".failed-archive-*"))

    archive = archive_failed_epoch(stored, state_path)
    archive_path = root / "epoch1-extrapolation_failed"
    assert archive["status"] == "archived" and archive_path.is_dir()
    assert (root / "epoch1" / "phase_diagram" / "INVALIDATED.md").is_file()
    assert (archive_path / "phase_diagram" / "INVALIDATED.md").is_file()
    assert validate_bundle(root / "latest" / "phase_diagram")["identity"]["hull_version"] == "old-hull"
    assert archive_failed_epoch(stored, state_path)["status"] == "already_archived"
    (archive_path / "phase_diagram" / "INVALIDATED.md").write_text("tampered", encoding="utf8")
    with pytest.raises(ValueError, match="existing_conflict"):
        archive_failed_epoch(stored, state_path)
    # A later automatic analysis must be recoverable even if it published a
    # different hull version from the same invalid model.
    publish_bundle(root, kind="phase_diagram", epoch="epoch1", model="bad",
                   identity={"hull_version": "republished-bad"},
                   required=["report.md"], build=build)
    stored["phase_diagrams"]["mlip"] = {"status": "completed", "version": "republished-bad", "model_version": "bad"}
    result = quarantine_refresh_publication(stored, state_path, "epoch_refresh")
    assert result["status"] == "quarantine_restored"
    assert validate_bundle(root / "latest" / "phase_diagram")["identity"]["hull_version"] == "old-hull"
    assert json.loads(state_path.read_text(encoding="utf8"))["phase_diagrams"]["mlip"] == old


def test_verified_mc_recovery_uses_previous_epoch_baseline(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import direct_extrapolation_sampling
    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    frame = tmp_path / "frame.vasp"
    frame.write_bytes(b"frame")
    sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    old = {"status": "completed", "version": "old-hull", "model_version": "old"}
    state = {"active_model_version": "bad", "phase_diagrams": {"mlip": {"version": "bad-hull", "model_version": "bad"}},
             "model_refresh": {"status": "completed", "old_model_version": "old", "new_model_version": "bad", "old_diagram": old},
             "relax_mc_batches": {"batch": {"formal_recovery_status": "accepted", "model_version": "bad"}},
             "tasks": [{"task_id": "mc", "model_version": "bad", "relax_mc_bundle_id": "batch"}],
             "direct_sampling_results": {"batch": {"model_version": "bad", "source_digest": "d", "selected_count": 1,
                 "selected": [{"candidate_id": "c", "source_task_id": "mc", "structure_path": str(frame), "structure_sha256": sha}]}}}
    updated, result = quarantine_refresh(state, "batch")
    assert result["status"] == "quarantined" and updated["phase_diagrams"]["mlip"] == old
    assert updated["extrapolation_recovery_cycle"]["invalidated_model_versions"] == ["bad"]


@pytest.mark.parametrize("fail", [False, True])
def test_finalization_automatically_quarantines_completed_direct(monkeypatch, fail):
    from phase_agent.tools.workflows.lifecycle_finalization import _finalize_workflow
    from phase_agent.tools.workflows import quarantine_refresh_publication as quarantine
    from phase_agent.tools.step_runner import file_protocol
    receipt = {"scope_id": "epoch_refresh", "source_digest": "digest"}
    state = {"direct_sampling_results": {"epoch_refresh": receipt}, "tasks": []}
    updated = {**state, "extrapolation_recovery_cycle": {"scope_id": "epoch_refresh", "direct_source_digest": "digest"}}
    monkeypatch.setattr("phase_agent.tools.workflows.lifecycle_finalization._save_runner_state", lambda *a: None)
    monkeypatch.setattr("phase_agent.analysis.feedback.export_dft_products.export_dft_products", lambda *a: None)
    monkeypatch.setattr("phase_agent.tools.remote.summarize_manual_upload_wait.summarize_manual_upload_wait", lambda *a, **k: None)
    def publish(actual, path, scope):
        assert actual is state and scope == "epoch_refresh"
        if fail:
            raise ValueError("unverified baseline")
        return {"status": "quarantined"}
    monkeypatch.setattr(quarantine, "quarantine_refresh_publication", publish)
    monkeypatch.setattr(file_protocol, "read_json", lambda *a: updated)
    frame = {"task_runner": None, "result": {"status": "completed", "state": state, "events": [{"execution": {"status": "completed", "result": {"direct_sampling": receipt}}}]}, "state_path": "unused", "effective_config": {}, "recovered_count": 0, "execution_mode": "interactive", "collection_report": None, "snapshot": {"config_version": "v"}}
    result = _finalize_workflow(frame)
    if fail:
        assert result["status"] == "not_configured"
        assert result["direct_quarantine"]["status"] == "failed"
        assert "unverified baseline" in result["reason"]
    else:
        assert result["direct_quarantine"]["status"] == "quarantined"
        assert result["state"] is updated


def test_switch_archives_previous_cycle_and_preserves_successful_records(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import direct_extrapolation_sampling, perturbation_direct
    from phase_agent.agents.reliability import extrapolation_round
    monkeypatch.setattr(direct_extrapolation_sampling, 'valid_direct_receipt', lambda receipt: True)
    monkeypatch.setattr(extrapolation_round, 'recovery_dft_failure_settled', lambda state, cycle: True)
    frame = tmp_path / 'mh1.vasp'
    frame.write_bytes(b'frame')
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    row = {'candidate_id':'p', 'source_task_id':'mh1', 'structure_path':str(frame), 'structure_sha256':digest}
    monkeypatch.setattr(perturbation_direct, 'perturbation_candidates', lambda *a, **kw: ([row], {}))
    old = {'status':'completed','version':'h','model_version':'old','entries':[]}
    prior = {'scope_id':'epoch_refresh','status':'awaiting_reviewed_dft_labels', 'invalid_model_version':'bad',
             'direct_source_digest':'prior','selected':[{'candidate_id':'old','structure_sha256':'old-sha'}]}
    state = {'active_model_version':'bad', 'phase_diagrams':{'mlip':old},
        'model_refresh':{'old_model_version':'old','new_model_version':'bad','old_diagram':old,'task_ids':['r']},
        'tasks':[{'task_id':'r','model_version':'bad'},{'task_id':'mh1','model_version':'mace-mh-1'}],
        'dft_dataset_records':[{'task_id':'previous-success'}], 'extrapolation_recovery_cycle':prior,
        'direct_sampling_results':{'epoch_refresh':{'model_version':'bad','recovery_route':'perturbation',
            'round_evidence_digest':'round','source_digest':'new','selected_count':1,'selected':[row]}}}
    updated, receipt = quarantine_refresh(state, 'epoch_refresh')
    assert receipt['status'] == 'quarantined'
    assert updated['extrapolation_recovery_history'][0]['status'] == 'superseded_by_perturbation'
    assert updated['extrapolation_recovery_history'][0]['selected'] == prior['selected']
    assert updated['dft_dataset_records'] == [{'task_id':'previous-success'}]
    assert updated['extrapolation_recovery_cycle']['selected'][0]['candidate_id'] == 'p'


def test_recovery_proposes_dft_for_exact_direct_selection(monkeypatch):
    from types import SimpleNamespace
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.tools.workflows import prepare_dft_proposal as preparation
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    monkeypatch.setattr(extrapolation_round, 'required_direct_round', lambda state: None)
    monkeypatch.setattr(extrapolation_recovery, 'recovery_context', lambda state: {'labels':{'status':'waiting','label_task_ids':[]}})
    monkeypatch.setattr(preparation, 'prepare_dft_proposal', lambda action, state, *a, **k: SimpleNamespace(status='ready', action=action, state=state))
    cycle = {'status':'awaiting_reviewed_dft_labels','invalid_model_version':'m','recovery_approval_record_id':'approved',
             'selected':[{'candidate_id':'picked','structure_sha256':'sha'}]}
    state = {'extrapolation_recovery_cycle':cycle,'tasks':[]}
    action = {'tool':'select_dft_candidates','task_key':'dft','parameters':{'decisions':[{'candidate_id':'picked','action':'DFT_SINGLE_POINT','reason':'representative'}]}}
    result = select_initial_proposal(current=state, mode='interactive', config={'agent':{'allowed_tools':['select_dft_candidates']}},
        context={}, decision_state={}, registry={'select_dft_candidates':{'handler':lambda **k:None}}, agent_client=lambda payload:None,
        invocation_id='i', propose=lambda *a, **k:action, revise=None, prepare_debug=None,
        is_verified_second=None, mc_block=None, model_failed=None, record_id_factory=lambda *a:'r')
    assert result['_proposal_selected'] and result['proposal']['raw_action']['tool'] == 'select_dft_candidates'
    action['parameters']['decisions'][0]['candidate_id'] = 'unrelated'
    rejected = select_initial_proposal(current=state, mode='interactive', config={'agent':{'allowed_tools':['select_dft_candidates']}},
        context={}, decision_state={}, registry={'select_dft_candidates':{'handler':lambda **k:None}}, agent_client=lambda payload:None,
        invocation_id='i', propose=lambda *a, **k:action, revise=None, prepare_debug=None,
        is_verified_second=None, mc_block=None, model_failed=None, record_id_factory=lambda *a:'r')
    assert rejected['status'] == 'not_configured' and not rejected['submitted']


@pytest.mark.parametrize("published_model,scope,expected", [
    ("bad", "epoch_refresh", "bad-hull"),
    ("other", "epoch_refresh", None),
    ("bad", "other_scope", None),
])
def test_invalidated_sentinel_requires_matching_published_hull(
    tmp_path, monkeypatch, published_model, scope, expected
):
    from phase_agent.analysis.feedback import versioned_products
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    monkeypatch.setattr(versioned_products, "validate_bundle", lambda path: {
        "identity": {"model_version": published_model, "hull_version": "bad-hull"}
    })
    frame = tmp_path / "frame.vasp"
    frame.write_bytes(b"frame")
    old = {"status": "completed", "version": "old-hull", "model_version": "old"}
    state = {
        "analysis_output_directory": str(tmp_path),
        "active_model_version": "bad",
        "phase_diagrams": {"mlip": {"status": "invalidated", "model_version": "bad",
            "reason": "large_extrapolation_failure", "invalidated_rounds": [scope]}},
        "model_refresh": {"old_model_version": "old", "new_model_version": "bad",
            "old_diagram": old, "task_ids": ["r1"]},
        "tasks": [{"task_id": "r1", "model_version": "bad"}],
        "direct_sampling_results": {"epoch_refresh": {"model_version": "bad",
            "source_digest": "digest", "selected_count": 1, "selected": [{
                "candidate_id": "c", "source_task_id": "r1", "structure_path": str(frame),
                "structure_sha256": hashlib.sha256(frame.read_bytes()).hexdigest()}]}},
    }
    if expected is None:
        with pytest.raises(ValueError, match="recovery_current_hull_unbound"):
            quarantine_refresh(state, "epoch_refresh")
    else:
        updated, receipt = quarantine_refresh(state, "epoch_refresh")
        assert receipt["invalid_hull_version"] == expected
        assert updated["phase_diagrams"]["mlip"] == old
