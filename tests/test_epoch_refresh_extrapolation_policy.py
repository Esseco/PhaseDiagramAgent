from phase_agent.agents.reliability.extrapolation_round import (
    assess_current_rounds,
    required_direct_round,
)
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram


def test_relax_mc_extrapolation_is_evidence_only():
    rows = [
        {
            "structure_id": str(i),
            "identity_scheme": "periodic_input_v1",
            "monitor_complete": True,
            "extrapolation_failure": i == 0,
        }
        for i in range(10)
    ]
    task = {
        "task_id": "mc",
        "model_version": "m",
        "stage": "deep_search",
        "status": "completed",
        "relax_mc_bundle_id": "b",
        "outputs": {
            "relaxation_observations": rows,
            "relaxation_observations_complete": True,
            "actual_mc_steps": 9,
        },
    }
    state = {
        "active_model_version": "m",
        "tasks": [task],
        "relax_mc_batches": {
            "b": {
                "model_version": "m",
                "status": "completed",
                "formal_recovery_status": "accepted",
                "import_checksum": "sha",
                "result_summary": {"task_count": 1},
            }
        },
    }
    facts = assess_current_rounds(state)["rounds"][0]
    assert facts["failure_fraction"] == 0.1
    assert facts["direct_sampling_required"] is False
    assert required_direct_round(state) is None
    result = update_phase_diagram([], active_model_version="m", output_state=state)
    assert (result["diagrams"].get("mlip") or {}).get("status") != "invalidated"


def test_sampling_route_and_bounded_progress():
    from phase_agent.evidence.extrapolation_policy import assess_policy

    for epoch in (0, 1):
        assert assess_policy(epoch, 0.1, 0, 0, False)["route"] == "perturbation"
    assert assess_policy(2, 0.099, 0, 0, False)["status"] == "clear"
    assert assess_policy(2, 0.1, 0, 0, False)["route"] == "perturbation"
    assert assess_policy(2, 0.199, 0, 0, False)["route"] == "perturbation"
    assert assess_policy(2, 0.2, 0, 0, False)["route"] == "perturbation"
    assert assess_policy(2, 0.1, 1, 5, True)["route"] == "perturbation"
    assert assess_policy(2, 0.1, 5, 49, False)["status"] == "attempt_limit"
    assert assess_policy(2, 0.1, 5, 50, False)["status"] == "complete"
    assert assess_policy(2, None, 0, 0, False)["status"] == "unknown"
    assert assess_policy(None, 0.1, 0, 0, False)["status"] == "sampling_required"


def test_recovery_progress_deduplicates_across_attempts(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    from phase_agent.science.dft import spin_acceptance

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda row: True)
    tasks, records, selected = [], [], []
    for i in range(50):
        path = tmp_path / f"{i}.vasp"
        path.write_text(str(i), encoding="utf8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks.append(
            {
                "task_id": f"t{i}",
                "object_id": f"c{i}",
                "model_version": "m",
                "input_structure_sha256": digest,
                "stage": "dft_single_point",
                "status": "completed",
                "structure_path": str(path),
            }
        )
        records.append(
            {
                "task_id": f"t{i}",
                "status": "completed",
                "converged": True,
                "checks_passed": True,
                "final_frame_valid": True,
                "training_ready": True,
            }
        )
        selected.append({"candidate_id": f"c{i}", "structure_sha256": digest})
    base = {
        "scope_id": "epoch_refresh",
        "invalid_model_version": "m",
        "policy_version": "epoch_refresh_v1",
        "round_evidence_digest": "bound",
        "status": "awaiting_reviewed_dft_labels",
    }
    prior = {**base, "direct_source_digest": "first", "selected": selected[:30]}
    cycle = {**base, "direct_source_digest": "second", "selected": selected[29:]}
    state = {
        "extrapolation_recovery_cycle": cycle,
        "extrapolation_recovery_history": [prior],
        "tasks": tasks,
        "dft_dataset_records": records,
    }
    facts = recovery_label_facts(state)
    assert facts["status"] == "complete" and facts["labelled_count"] == 50
    assert facts["attempt_count"] == 2 and len(facts["label_task_ids"]) == 50
    state["tasks"][-1]["model_version"] = "old-model"
    assert recovery_label_facts(state)["labelled_count"] == 49
    state["tasks"][-1]["model_version"] = "m"
    state["tasks"][-1]["object_id"] = "unrelated"
    assert recovery_label_facts(state)["labelled_count"] == 49
    state["tasks"][-1]["object_id"] = "c49"
    state["dft_dataset_records"].pop()
    facts = recovery_label_facts(state)
    assert facts["status"] == "waiting" and facts["labelled_count"] == 49


def test_finished_attempt_can_continue_without_retry(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
    from phase_agent.science.dft import spin_acceptance

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda row: True)
    p = tmp_path / "frame.vasp"
    p.write_text("frame", encoding="utf8")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    cycle = {
        "scope_id": "epoch_refresh",
        "invalid_model_version": "m",
        "policy_version": "epoch_refresh_v1",
        "round_evidence_digest": "bound",
        "direct_source_digest": "attempt",
        "selected": [{"candidate_id": "c", "structure_sha256": digest}],
    }
    task = {
        "task_id": "t",
        "object_id": "c",
        "model_version": "m",
        "input_structure_sha256": digest,
        "stage": "dft_single_point",
        "status": "completed",
        "structure_path": str(p),
    }
    record = {
        "task_id": "t",
        "status": "completed",
        "converged": True,
        "checks_passed": True,
        "final_frame_valid": True,
        "training_ready": True,
    }
    state = {
        "extrapolation_recovery_cycle": cycle,
        "tasks": [task],
        "dft_dataset_records": [record],
    }
    assert recovery_dft_failure_settled(state, cycle)
    state["dft_dataset_records"] = []
    assert not recovery_dft_failure_settled(state, cycle)
    state["dft_dataset_records"] = [record]
    task["status"] = "running"
    assert not recovery_dft_failure_settled(state, cycle)


def test_new_epoch_quarantine_uses_bounded_policy(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    frame = tmp_path / "frame.vasp"
    frame.write_text("x", encoding="utf8")
    old = {"status": "completed", "version": "old", "model_version": "old"}
    state = {
        "active_model_version": "m",
        "upload_layout": {"model_rounds": {"m": 3}},
        "phase_diagrams": {"mlip": {"status": "completed", "version": "new", "model_version": "m"}},
        "model_refresh": {
            "old_model_version": "old",
            "new_model_version": "m",
            "old_diagram": old,
            "task_ids": ["r"],
        },
        "tasks": [{"task_id": "r", "model_version": "m"}],
        "direct_sampling_results": {
            "epoch_refresh": {
                "model_version": "m",
                "source_digest": "digest",
                "round_evidence_digest": "bound",
                "policy_version": "epoch_refresh_v1",
                "selected_count": 1,
                "selected": [
                    {
                        "candidate_id": "c",
                        "source_task_id": "r",
                        "structure_path": str(frame),
                        "structure_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    updated, _ = quarantine_refresh(state, "epoch_refresh")
    assert updated["extrapolation_recovery_cycle"]["policy_version"] == "epoch_refresh_v1"


def test_controller_enforces_route_and_limit(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    monkeypatch.setattr(
        extrapolation_round,
        "required_direct_round",
        lambda state: {
            "scope_id": "epoch_refresh",
            "evidence_digest": "bound",
            "failure_fraction": 0.1,
        },
    )
    monkeypatch.setattr(extrapolation_round, "recovery_dft_failure_settled", lambda *a: True)
    monkeypatch.setattr(
        extrapolation_recovery,
        "recovery_label_facts",
        lambda state: {"status": "waiting", "attempt_count": 5, "labelled_count": 12},
    )
    action = {
        "operation": "sample_direct_extrapolation",
        "tool": "prepare_local_batch_files",
        "parameters": {
            "mode": "direct_extrapolation",
            "scope_id": "epoch_refresh",
            "round_evidence_digest": "bound",
        },
        "task_key": "t",
        "budget": 0,
        "reason": "sampling",
        "expected_purpose": "test",
    }

    def run(state):
        return select_initial_proposal(
            current=state,
            mode="interactive",
            config={"agent": {"allowed_tools": ["prepare_local_batch_files"]}},
            context={},
            decision_state={},
            registry={"prepare_local_batch_files": {"handler": lambda **kw: None}},
            agent_client=lambda x: None,
            invocation_id="t",
            propose=lambda *a, **k: action,
            revise=None,
            prepare_debug=None,
            is_verified_second=None,
            mc_block=None,
            model_failed=None,
            record_id_factory=lambda *a: "r",
        )

    state = {"active_model_version": "m", "upload_layout": {"model_rounds": {"m": 3}}}
    proposed = run(state)
    assert proposed["status"] == "not_configured"
    action["operation"] = "prepare_parent_first_recovery"
    action["parameters"]["mode"] = "parent_first_recovery_inputs"
    assert run(state).get("_proposal_selected") is True
    action["operation"] = "sample_direct_extrapolation"
    action["parameters"]["mode"] = "direct_extrapolation"
    state["upload_layout"]["model_rounds"]["m"] = 1
    assert run(state)["status"] == "not_configured"
    state["upload_layout"]["model_rounds"]["m"] = 3
    state["extrapolation_recovery_cycle"] = {
        "policy_version": "epoch_refresh_v1",
        "scope_id": "epoch_refresh",
        "invalid_model_version": "m",
        "round_evidence_digest": "bound",
        "recovery_route": "extrapolation",
    }
    assert run(state)["extrapolation_policy"]["status"] == "attempt_limit"
    state["extrapolation_recovery_cycle"]["round_evidence_digest"] = "changed"
    assert run(state)["extrapolation_policy"]["status"] == "unknown"


def test_two_verified_attempts_accumulate_and_do_not_retry(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh, recovery_label_facts
    from phase_agent.science.dft import spin_acceptance
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda value: True)
    old = {"status": "completed", "version": "old", "model_version": "old"}
    state = {
        "active_model_version": "m",
        "upload_layout": {"model_rounds": {"m": 3}},
        "phase_diagrams": {"mlip": {"status": "completed", "version": "new", "model_version": "m"}},
        "model_refresh": {
            "old_model_version": "old",
            "new_model_version": "m",
            "old_diagram": old,
            "task_ids": ["refresh"],
        },
        "tasks": [{"task_id": "refresh", "model_version": "m"}],
        "dft_dataset_records": [],
    }
    for attempt, count in enumerate((20, 30)):
        selected = []
        for i in range(count):
            idx = attempt * 100 + i
            path = tmp_path / f"{idx}.vasp"
            path.write_text(str(idx), encoding="utf8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            selected.append(
                {
                    "candidate_id": f"c{idx}",
                    "source_task_id": "refresh",
                    "structure_path": str(path),
                    "structure_sha256": digest,
                }
            )
        state["direct_sampling_results"] = {
            "epoch_refresh": {
                "model_version": "m",
                "source_digest": f"att-{attempt}",
                "round_evidence_digest": "bound",
                "policy_version": "epoch_refresh_v1",
                "selected_count": count,
                "selected": selected,
            }
        }
        state, _ = quarantine_refresh(state, "epoch_refresh")
        for i, row in enumerate(selected):
            tid = f"d{attempt}-{i}"
            state["tasks"].append(
                {
                    "task_id": tid,
                    "object_id": row["candidate_id"],
                    "input_structure_sha256": row["structure_sha256"],
                    "stage": "dft_single_point",
                    "status": "completed",
                    "model_version": "m",
                    "structure_path": row["structure_path"],
                }
            )
            state["dft_dataset_records"].append(
                {
                    "task_id": tid,
                    "status": "completed",
                    "converged": True,
                    "checks_passed": True,
                    "final_frame_valid": True,
                    "training_ready": True,
                }
            )
    facts = recovery_label_facts(state)
    assert facts["status"] == "complete" and facts["labelled_count"] == 50
    assert facts["attempt_count"] == 2
    assert state["phase_diagrams"]["mlip"] == old


def test_new_direct_result_has_informative_name_and_is_idempotent(tmp_path, monkeypatch):
    from tests.test_direct_extrapolation_sampling import make_case
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.workflows.direct_extrapolation_sampling import prepare_direct_sampling
    from Process_MLIP.direct_sampling import workflow

    state, context, action, _ = make_case(tmp_path, monkeypatch)
    state["upload_layout"] = {"model_rounds": {"m": 3}}
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda s: {
            "model_version": "m",
            "rounds": [
                {
                    "scope_id": "epoch_refresh",
                    "evidence_digest": "bound",
                    "failure_fraction": 0.1,
                    "direct_sampling_required": True,
                }
            ],
        },
    )
    action["parameters"].update(
        round_evidence_digest="bound", extrapolation_policy_version="epoch_refresh_v1"
    )
    first = prepare_direct_sampling(action=action, context=context)
    assert "attempt_01_extrapolation_" in first["direct_sampling"]["plan_path"]
    assert first["direct_sampling"]["policy_version"] == "epoch_refresh_v1"
    monkeypatch.setattr(
        workflow, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("repeated sampling"))
    )
    again = prepare_direct_sampling(
        action=action, context={**context, "event_state": first["state"]}
    )
    assert again["direct_sampling"]["source_digest"] == first["direct_sampling"]["source_digest"]


def test_parent_first_new_policy_keeps_partial_labels_for_next_attempt(tmp_path, monkeypatch):
    from tests.test_parent_first_training_recovery import make_state
    from phase_agent.evidence.parent_first_recovery import advance_parent_first_recovery
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts

    state = make_state(tmp_path, monkeypatch, 5)
    for task in state["tasks"]:
        task["object_id"] = task["structure_id"]
    state["parent_first_recovery_batches"]["batch"]["policy_version"] = "epoch_refresh_v1"
    updated = advance_parent_first_recovery(state)
    facts = recovery_label_facts(updated)
    assert updated["parent_first_recovery_batches"]["batch"]["status"] == "results_received"
    assert facts["status"] == "waiting" and facts["labelled_count"] == 4
    assert facts["attempt_count"] == 1


def test_recovery_dft_failure_requires_analysis(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.dft_failure_policy import completed_failure_batches
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    from phase_agent.science.dft import spin_acceptance
    from tests.test_dft_failure_policy import make_state

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda row: True)
    state, recovered = make_state(1, 1)
    selected = []
    for task in state["tasks"]:
        p = tmp_path / (task["task_id"] + ".vasp")
        p.write_text(task["task_id"], encoding="utf8")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        task["structure_path"] = str(p)
        task["object_id"] = task["task_id"]
        task["input_structure_sha256"] = digest
        selected.append({"candidate_id": task["task_id"], "structure_sha256": digest})
    state.update(
        active_model_version="m",
        extrapolation_recovery_cycle={
            "policy_version": "epoch_refresh_v1",
            "scope_id": "epoch_refresh",
            "invalid_model_version": "m",
            "direct_source_digest": "digest",
            "round_evidence_digest": "bound",
            "selected": selected,
        },
    )
    assert completed_failure_batches(state, recovered)[0]["failed_count"] == 1
    facts = recovery_label_facts(state)
    assert facts["failed_count"] == 1 and facts["failure_groups"][0]["reason"] == "SCF"
    state["tasks"][0]["structure_path"] = str(tmp_path / "missing")
    assert completed_failure_batches(state, recovered)


def test_two_parent_first_attempts_reach_training_threshold(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    from phase_agent.evidence.parent_first_recovery import advance_parent_first_recovery
    from phase_agent.science.dft import spin_acceptance
    from phase_agent.tools.workflows import parent_first_result

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda row: True)
    selections = {}
    monkeypatch.setattr(
        parent_first_result,
        "validate_parent_first_result",
        lambda input_dir, selection_dir: {
            "target": len(selections[input_dir]),
            "result_sha256": input_dir,
            "selected": selections[input_dir],
        },
    )
    baseline = {"status": "completed", "version": "old", "model_version": "old"}
    state = {
        "active_model_version": "m",
        "tasks": [],
        "dft_dataset_records": [],
        "parent_first_recovery_batches": {},
        "phase_diagrams": {"mlip": baseline},
        "model_refresh": {
            "new_model_version": "m",
            "old_diagram": baseline,
            "task_ids": ["refresh"],
        },
    }
    for attempt, size in enumerate((20, 32)):
        batch_id = f"b{attempt}"
        selected, task_ids = [], []
        for i in range(size):
            sid = f"s{attempt}-{i}"
            path = tmp_path / (sid + ".vasp")
            path.write_text(sid, encoding="utf8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            tid = f"t{attempt}-{i}"
            task_ids.append(tid)
            selected.append({"structure_sha256": digest})
            state["tasks"].append(
                {
                    "task_id": tid,
                    "structure_id": sid,
                    "object_id": sid,
                    "stage": "dft_single_point",
                    "status": "failed" if i == 0 else "completed",
                    "model_version": "m",
                    "structure_path": str(path),
                    "input_structure_sha256": digest,
                }
            )
            if i:
                state["dft_dataset_records"].append(
                    {
                        "task_id": tid,
                        "status": "completed",
                        "converged": True,
                        "checks_passed": True,
                        "final_frame_valid": True,
                        "training_ready": True,
                    }
                )
        selections[batch_id] = selected
        state["parent_first_recovery_batches"][batch_id] = {
            "status": "waiting_dft_results",
            "policy_version": "epoch_refresh_v1",
            "target": size,
            "scope_id": "epoch_refresh",
            "model_version": "m",
            "round_evidence_digest": "bound",
            "approval_record_id": f"a{attempt}",
            "input_dir": batch_id,
            "selection_dir": "selected",
            "sampling_result_sha256": batch_id,
            "dft_task_ids": task_ids,
        }
        state = advance_parent_first_recovery(state)
    facts = recovery_label_facts(state)
    assert facts["status"] == "complete" and facts["labelled_count"] == 50
    assert facts["attempt_count"] == 2 and facts["failed_count"] == 2
    assert len(state["extrapolation_recovery_history"]) == 1
    from phase_agent.evidence.extrapolation_recovery import recovery_model_matches_labels

    data_ids = [tid + ":frame:0" for tid in facts["label_task_ids"]]
    state["remote_finetune_jobs"] = {
        "job": {
            "status": "results_received",
            "original_model_version": "m",
            "training_handoff": {"candidate_model_version": "next"},
            "report": {"dataset_scope": {"data_ids": data_ids}},
        }
    }
    assert recovery_model_matches_labels(state, "next")
    data_ids.pop()
    assert not recovery_model_matches_labels(state, "next")


def test_changed_or_missing_refresh_evidence_cannot_clear_recovery(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round

    state = {
        "active_model_version": "m",
        "extrapolation_recovery_cycle": {
            "policy_version": "epoch_refresh_v1",
            "status": "awaiting_reviewed_dft_labels",
            "scope_id": "epoch_refresh",
            "invalid_model_version": "m",
            "round_evidence_digest": "old",
        },
    }
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda state: {
            "model_version": "m",
            "rounds": [
                {
                    "scope_id": "epoch_refresh",
                    "evidence_digest": "changed",
                    "direct_sampling_required": None,
                }
            ],
        },
    )
    assert extrapolation_round.required_direct_round(state)["evidence_digest"] == "changed"
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda state: {"model_version": "m", "rounds": []},
    )
    assert extrapolation_round.required_direct_round(state)["evidence_digest"] is None


def test_parent_first_handoff_obeys_policy_and_names_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.local.prepare_parent_first_recovery import prepare_parent_first_recovery
    from phase_agent.tools.workflows import parent_first_submission

    monkeypatch.setattr(
        extrapolation_round,
        "required_direct_round",
        lambda state: {
            "scope_id": "epoch_refresh",
            "evidence_digest": "bound",
            "failure_fraction": 0.2,
        },
    )
    monkeypatch.setattr(
        parent_first_submission,
        "prepare_parent_first_submission",
        lambda state, ledger, params, target, **kw: {
            "status": "prepared",
            "submitted": False,
            "parent_count": 1,
            "reference_count": 0,
            "target": 40,
            "task_id": "t",
            "input_dir": str(target),
            "handoff": {},
        },
    )
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "upload_layout": {"model_rounds": {"m": 1}},
    }
    params = {
        "mode": "parent_first_recovery_inputs",
        "confirmed": True,
        "scope_id": "epoch_refresh",
        "round_evidence_digest": "bound",
        "extrapolation_policy_version": "epoch_refresh_v1",
        "dft_parameters": {},
    }
    result = prepare_parent_first_recovery(
        action={"operation": "prepare_parent_first_recovery", "parameters": params},
        context={
            "event_state": state,
            "effective_config": {"upload_batches_directory": str(tmp_path), "dft": {}},
            "manager": SimpleNamespace(data={}),
            "approval_record_id": "approved",
            "config_version": "c",
        },
    )
    assert result["status"] == "prepared"
    assert (
        "extrapolation-round-01-perturbation-attempt-01-"
        in result["parent_first_recovery_batch"]["input_dir"]
    )
    assert result["parent_first_recovery_batch"]["policy_version"] == "epoch_refresh_v1"
    state["upload_layout"]["model_rounds"]["m"] = 3
    monkeypatch.setattr(
        extrapolation_round,
        "required_direct_round",
        lambda state: {
            "scope_id": "epoch_refresh",
            "evidence_digest": "bound",
            "failure_fraction": 0.1,
        },
    )
    assert (
        prepare_parent_first_recovery(
            action={"operation": "prepare_parent_first_recovery", "parameters": params},
            context={
                "event_state": state,
                "effective_config": {"upload_batches_directory": str(tmp_path), "dft": {}},
                "manager": SimpleNamespace(data={}),
                "approval_record_id": "approved",
                "config_version": "c",
            },
        )["status"]
        == "prepared"
    )


def test_failed_replacement_refresh_starts_new_epoch_without_losing_history(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda value: True)
    frame = tmp_path / "new.vasp"
    frame.write_text("new", encoding="utf8")
    old_hull = {"status": "completed", "version": "baseline", "model_version": "baseline-model"}
    prior = {
        "status": "awaiting_new_model_refresh",
        "scope_id": "epoch_refresh",
        "invalid_model_version": "failed-epoch1",
        "replacement_model_version": "failed-epoch2",
        "baseline_diagram": old_hull,
        "baseline_model_version": "baseline-model",
        "baseline_hull_version": "baseline",
        "policy_version": "epoch_refresh_v1",
        "recovery_lineage_id": "lineage",
        "extrapolation_round": 1,
        "origin_epoch": "epoch1",
        "round_evidence_digest": "prior-evidence",
        "direct_source_digest": "prior",
        "selected": [{"candidate_id": "old", "structure_sha256": "old-hash"}],
    }
    receipt = {
        "model_version": "failed-epoch2",
        "source_digest": "new-attempt",
        "round_evidence_digest": "new-evidence",
        "policy_version": "epoch_refresh_v1",
        "selected_count": 1,
        "selected": [
            {
                "candidate_id": "new",
                "source_task_id": "r",
                "structure_path": str(frame),
                "structure_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            }
        ],
    }
    state = {
        "active_model_version": "failed-epoch2",
        "extrapolation_recovery_cycle": prior,
        "phase_diagrams": {"mlip": old_hull},
        "model_refresh": {
            "old_model_version": "baseline-model",
            "new_model_version": "failed-epoch2",
            "old_diagram": old_hull,
            "task_ids": ["r"],
        },
        "tasks": [{"task_id": "r", "model_version": "failed-epoch2"}],
        "direct_sampling_results": {"epoch_refresh": receipt},
    }
    updated, _ = quarantine_refresh(state, "epoch_refresh")
    assert updated["extrapolation_recovery_history"][-1]["status"] == "replacement_refresh_failed"
    assert updated["extrapolation_recovery_cycle"]["invalid_model_version"] == "failed-epoch2"
    assert updated["extrapolation_recovery_cycle"]["round_evidence_digest"] == "new-evidence"
    assert updated["extrapolation_recovery_cycle"]["recovery_lineage_id"] == "lineage"
    assert updated["extrapolation_recovery_cycle"]["extrapolation_round"] == 2
    assert updated["extrapolation_recovery_cycle"]["origin_epoch"] == "epoch1"
    assert updated["phase_diagrams"]["mlip"] == old_hull


def test_parent_first_next_epoch_restarts_after_failed_refresh(tmp_path, monkeypatch):
    from tests.test_parent_first_training_recovery import make_state
    from phase_agent.evidence.parent_first_recovery import advance_parent_first_recovery
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts

    state = make_state(tmp_path, monkeypatch, 5)
    baseline = state["model_refresh"]["old_diagram"]
    for task in state["tasks"]:
        task["object_id"] = task["structure_id"]
    state["parent_first_recovery_batches"]["batch"]["policy_version"] = "epoch_refresh_v1"
    state["extrapolation_recovery_cycle"] = {
        "status": "awaiting_new_model_refresh",
        "scope_id": "epoch_refresh",
        "invalid_model_version": "previous",
        "replacement_model_version": "bad",
        "baseline_diagram": baseline,
        "baseline_model_version": baseline["model_version"],
        "baseline_hull_version": baseline["version"],
        "policy_version": "epoch_refresh_v1",
        "round_evidence_digest": "previous-evidence",
        "direct_source_digest": "previous-attempt",
        "selected": [{"candidate_id": "old", "structure_sha256": "old"}],
    }
    updated = advance_parent_first_recovery(state)
    assert updated["extrapolation_recovery_history"][-1]["status"] == "replacement_refresh_failed"
    assert updated["extrapolation_recovery_cycle"]["invalid_model_version"] == "bad"
    assert recovery_label_facts(updated)["attempt_count"] == 1


def test_new_model_refresh_advances_epoch_and_resets_recovery_identity(monkeypatch):
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh

    monkeypatch.setattr(
        extrapolation_recovery, "recovery_label_facts", lambda state: {"status": "complete"}
    )
    baseline = {"status": "completed", "version": "h0", "model_version": "m0"}
    state = {
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2}},
        "phase_diagrams": {"mlip": baseline},
        "extrapolation_recovery_cycle": {
            "status": "awaiting_reviewed_dft_labels",
            "invalid_model_version": "m1",
            "policy_version": "epoch_refresh_v1",
            "extrapolation_round": 1,
            "origin_epoch": "epoch1",
            "baseline_diagram": baseline,
            "baseline_hull_version": "h0",
            "baseline_model_version": "m0",
        },
    }
    require_structure_refresh(state, "m1", "m2")
    assert model_epoch(state, "m1") == "epoch1" and model_epoch(state, "m2") == "epoch2"
    assert state["upload_layout"]["model_rounds"]["m2"] == 3
    assert state["model_refresh"]["new_model_version"] == "m2"
    assert state["extrapolation_recovery_cycle"]["status"] == "awaiting_new_model_refresh"
    from phase_agent.tools.remote.model_upload_directory import model_upload_directory

    assert model_upload_directory("root", state, "m2").name == "epoch2"
    state["extrapolation_recovery_cycle"]["status"] = "completed"
    require_structure_refresh(state, "m2", "m3")
    assert model_epoch(state, "m3") == "epoch3"
    # A second recovery keeps the original lineage but advances from the latest model.
    state["extrapolation_recovery_cycle"].update(
        status="awaiting_reviewed_dft_labels", invalid_model_version="m3"
    )
    require_structure_refresh(state, "m3", "m4")
    assert model_epoch(state, "m4") == "epoch4"
    assert state["extrapolation_recovery_cycle"]["origin_epoch"] == "epoch1"


def test_extrapolation_round_keeps_five_attempt_limit_across_model_updates(monkeypatch):
    from phase_agent.evidence import extrapolation_recovery

    previous = {
        "policy_version": "epoch_refresh_v1",
        "status": "awaiting_new_model_refresh",
        "scope_id": "epoch_refresh",
        "invalid_model_version": "m1",
        "replacement_model_version": "m2",
        "round_evidence_digest": "old",
        "direct_source_digest": "d1",
        "selected": [{"structure_sha256": "x"}],
    }
    state = {
        "active_model_version": "m2",
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2, "m2": 3}},
        "extrapolation_recovery_cycle": previous,
    }
    round_facts = {"scope_id": "epoch_refresh", "evidence_digest": "new", "failure_fraction": 0.1}
    monkeypatch.setattr(
        extrapolation_recovery,
        "recovery_label_facts",
        lambda state: {"status": "complete", "attempt_count": 5, "labelled_count": 50},
    )
    assert (
        extrapolation_recovery.epoch_recovery_policy(state, round_facts)["status"]
        == "attempt_limit"
    )
    monkeypatch.setattr(
        extrapolation_recovery,
        "recovery_label_facts",
        lambda state: {"status": "complete", "attempt_count": 1, "labelled_count": 50},
    )
    assert (
        extrapolation_recovery.epoch_recovery_policy(state, round_facts)["route"] == "perturbation"
    )


def test_sampling_rule_uses_actual_model_updates_not_epoch_name():
    from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy

    state = {
        "active_model_version": "m2",
        "upload_layout": {
            "model_rounds": {"m0": 1, "m1": 2, "m2": 3},
            "scientific_epochs": {"m2": 0},
        },
    }
    facts = {"scope_id": "epoch_refresh", "evidence_digest": "bound", "failure_fraction": 0.1}
    assert epoch_recovery_policy(state, facts)["route"] == "perturbation"
    state["active_model_version"] = "m1"
    assert epoch_recovery_policy(state, facts)["route"] == "perturbation"


def test_shared_epoch_manifest_keeps_all_model_versions(tmp_path):
    import json
    from phase_agent.tools.remote.submission_architecture import write_epoch_manifests

    root = tmp_path / "submissions"
    (root / "epoch0").mkdir(parents=True)
    write_epoch_manifests(root, {"m0": 1, "m1": 2}, [], scientific_epochs={"m1": 0})
    manifest = json.loads((root / "epoch0" / "layout_manifest.json").read_text(encoding="utf8"))
    assert manifest["model_version"] == "m1" and manifest["model_versions"] == ["m0", "m1"]
    assert not (root / "epoch1").exists()


def test_conflicting_scientific_epoch_does_not_mutate_recovery(monkeypatch):
    from copy import deepcopy
    import pytest
    from phase_agent.evidence import extrapolation_recovery
    from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh

    monkeypatch.setattr(
        extrapolation_recovery, "recovery_label_facts", lambda state: {"status": "complete"}
    )
    baseline = {"status": "completed", "version": "h0", "model_version": "m0"}
    state = {
        "upload_layout": {
            "model_rounds": {"m0": 1, "m1": 2, "m2": 3},
            "scientific_epochs": {"m2": 99},
        },
        "phase_diagrams": {"mlip": baseline},
        "extrapolation_recovery_cycle": {
            "status": "awaiting_reviewed_dft_labels",
            "policy_version": "epoch_refresh_v1",
            "origin_epoch": "epoch1",
            "invalid_model_version": "m1",
            "baseline_diagram": baseline,
            "baseline_hull_version": "h0",
            "baseline_model_version": "m0",
        },
    }
    original = deepcopy(state)
    with pytest.raises(ValueError, match="scientific_epoch_conflict"):
        require_structure_refresh(state, "m1", "m2")
    assert state == original


def test_recovery_refresh_applies_hd_and_all_far(tmp_path, monkeypatch):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_state import refresh_preview

    state, _, context, _ = setup(tmp_path)
    baseline = state["model_refresh"]["old_diagram"]
    baseline.update(status="completed", version="hull-before-failure", model_version="old")
    baseline["entries"][0].update(is_stable=True, ehull=0.0, ehull_unit="eV/atom")
    manager = context["manager"]
    original = __import__("pathlib").Path(manager.data["structures"]["s"]["source_path"])
    for name in ("far", "missing"):
        path = tmp_path / (name + ".vasp")
        path.write_bytes(original.read_bytes() + ("\n" + name).encode())
        manager.data["structures"][name] = {**manager.data["structures"]["s"], "source_path": str(path)}
    baseline["entries"].append({"structure_id": "far", "structure_path": manager.data["structures"]["far"]["source_path"],
                                "ehull": 0.2, "ehull_unit": "eV/atom", "phase": "O3"})
    state["mlip_energy_policy"] = "latest_valid_per_structure_v1"
    state["extrapolation_recovery_cycle"] = {
        "status": "awaiting_new_model_refresh",
        "policy_version": "epoch_refresh_v1",
        "replacement_model_version": "new",
        "baseline_diagram": baseline,
        "baseline_model_version": "old",
        "baseline_hull_version": "hull-before-failure",
    }
    from phase_agent.tools.workflows import model_refresh_state
    from phase_agent.analysis.phase import phase_quality_reports
    monkeypatch.setattr(phase_quality_reports, "get_phase_quality_report",
                        lambda *a, **kw: {"snapshot_evidence_digest": "verified", "screening_policy": {}})
    baseline["entries"][0]["record_id"] = "base"
    baseline["entries"][1]["record_id"] = "far"
    monkeypatch.setattr(model_refresh_state, "_hd_reference",
                        lambda *a: ({"far"}, {"base": 0.0}))
    plan = refresh_preview(state, context["manager"], context["effective_config"])
    assert plan["baseline_hull_version"] == "hull-before-failure"
    assert plan["new_model_version"] == "new"
    assert {(row["structure_id"], row["operation"]) for row in plan["candidates"]} == {
        ("s", "relax"), ("far", "predict"), ("missing", "predict")
    }
    assert plan["refresh_scope"] == "trusted_baseline_hd_and_all_far"
    assert plan["policy"]["audit_fraction"] == 1.0
    assert any(r["structure_id"] == "far" and r["selection_reason"] == "high_hd" for r in plan["candidates"])


def test_replacement_refresh_exits_only_with_proven_fraction_below_ten_percent(
    tmp_path, monkeypatch
):
    from tests.test_model_refresh_workflow import recovered_state
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    state, _, _, _ = recovered_state(tmp_path)
    task = state["tasks"][0]
    source_sha = task["input_structure_sha256"]
    task.update(converged=True)
    task["outputs"].update(
        optimizer_converged=True,
        single_point_completed=False,
        refresh_force_followup={
            "input_structure_sha256": source_sha,
            "force_max_ev_per_angstrom": 0.11,
            "threshold_ev_per_angstrom": 0.1,
            "relaxed": True,
            "initial_prediction": {
                "single_point_completed": True,
                "forces_unit": "eV/angstrom",
                "forces": [[0.11, 0, 0]] * 4,
                "structure_checksum": source_sha,
                "mlip_version": "new",
            },
        },
    )
    state["extrapolation_recovery_cycle"] = {
        "policy_version": "epoch_refresh_v1",
        "status": "awaiting_new_model_refresh",
        "replacement_model_version": "new",
    }
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda current: {
            "rounds": [{"scope_id": "epoch_refresh", "status": "unknown", "failure_fraction": None}]
        },
    )
    updated, waiting = advance_refresh(state)
    assert (
        waiting
        and updated["extrapolation_recovery_cycle"]["status"] == "awaiting_new_model_refresh"
    )
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda current: {
            "rounds": [
                {"scope_id": "epoch_refresh", "status": "complete", "failure_fraction": 0.10}
            ]
        },
    )
    updated, waiting = advance_refresh(state)
    assert (
        waiting
        and updated["extrapolation_recovery_cycle"]["status"] == "awaiting_new_model_refresh"
    )
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda current: {
            "rounds": [
                {"scope_id": "epoch_refresh", "status": "complete", "failure_fraction": 0.09}
            ]
        },
    )
    updated, waiting = advance_refresh(state)
    assert not waiting and updated["extrapolation_recovery_cycle"]["status"] == "completed"


def test_attempt_counter_spans_extrapolation_rounds_but_labels_reset(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.evidence.extrapolation_recovery import (
        recovery_label_facts,
        epoch_recovery_policy,
    )

    history = [
        {
            "policy_version": "epoch_refresh_v1",
            "recovery_lineage_id": "lineage",
            "scope_id": "epoch_refresh",
            "invalid_model_version": "m1",
            "round_evidence_digest": "old",
            "direct_source_digest": f"old-{i}",
            "selected": [{"structure_sha256": f"old-hash-{i}"}],
        }
        for i in range(4)
    ]
    cycle = {
        "policy_version": "epoch_refresh_v1",
        "recovery_lineage_id": "lineage",
        "scope_id": "epoch_refresh",
        "invalid_model_version": "m2",
        "round_evidence_digest": "new",
        "direct_source_digest": "new-1",
        "selected": [{"candidate_id": "new-candidate", "structure_sha256": "new-hash"}],
    }
    state = {
        "active_model_version": "m2",
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2, "m2": 3}},
        "extrapolation_recovery_history": history,
        "extrapolation_recovery_cycle": cycle,
    }
    facts = recovery_label_facts(state)
    assert (
        facts["attempt_count"] == 5
        and facts["round_attempt_count"] == 1
        and facts["labelled_count"] == 0
    )
    monkeypatch.setattr(extrapolation_round, "recovery_dft_failure_settled", lambda *a: True)
    policy = epoch_recovery_policy(
        state, {"scope_id": "epoch_refresh", "evidence_digest": "new", "failure_fraction": 0.15}
    )
    assert policy["status"] == "attempt_limit"


def test_round_quarantine_marks_epoch_latest_and_keeps_distinct_notice(tmp_path, monkeypatch):
    import hashlib
    import json
    from phase_agent.analysis.feedback.versioned_products import publish_bundle, validate_bundle
    from phase_agent.tools.workflows import direct_extrapolation_sampling
    from phase_agent.tools.workflows.quarantine_refresh_publication import (
        quarantine_refresh_publication,
    )

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda receipt: True)
    project = tmp_path / "project"
    root = project / "analysis_outputs"
    state_path = project / "workflow_state" / "state.json"
    state_path.parent.mkdir(parents=True)

    def build(path):
        (path / "report.md").write_text("data", encoding="utf8")

    publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch0",
        model="old",
        identity={"hull_version": "old-hull"},
        required=["report.md"],
        build=build,
    )
    publish_bundle(
        root,
        kind="phase_diagram",
        epoch="epoch1",
        model="bad",
        identity={"hull_version": "bad-hull"},
        required=["report.md"],
        build=build,
    )
    frame = project / "frame.vasp"
    frame.write_text("frame", encoding="utf8")
    baseline = {"status": "completed", "version": "old-hull", "model_version": "old", "entries": []}
    state = {
        "analysis_output_directory": str(root),
        "active_model_version": "bad",
        "upload_layout": {"model_rounds": {"old": 1, "bad": 2}},
        "phase_diagrams": {
            "mlip": {"status": "completed", "version": "bad-hull", "model_version": "bad"}
        },
        "model_refresh": {
            "old_model_version": "old",
            "new_model_version": "bad",
            "old_diagram": baseline,
            "task_ids": ["r"],
        },
        "tasks": [{"task_id": "r", "model_version": "bad"}],
        "direct_sampling_results": {
            "epoch_refresh": {
                "model_version": "bad",
                "source_digest": "digest",
                "round_evidence_digest": "bound",
                "policy_version": "epoch_refresh_v1",
                "selected_count": 1,
                "selected": [
                    {
                        "candidate_id": "c",
                        "source_task_id": "r",
                        "structure_path": str(frame),
                        "structure_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf8")
    quarantine_refresh_publication(state, state_path, "epoch_refresh")
    assert (
        validate_bundle(root / "latest" / "phase_diagram")["identity"]["hull_version"] == "old-hull"
    )
    assert (
        json.loads(
            (root / "epoch1" / "phase_diagram" / "latest_status.json").read_text(encoding="utf8")
        )["status"]
        == "invalidated"
    )
    assert (
        len(
            list(
                (root / "epoch1" / "phase_diagram").glob("extrapolation_round_01_*_INVALIDATED.md")
            )
        )
        == 1
    )


def test_fifty_labels_still_waits_until_every_dft_returns(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    from phase_agent.science.dft import spin_acceptance

    monkeypatch.setattr(spin_acceptance, "spin_standard_passed", lambda row: True)
    tasks, records, selected = [], [], []
    for i in range(51):
        p = tmp_path / f"{i}.vasp"
        p.write_text(str(i), encoding="utf8")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        selected.append({"candidate_id": f"c{i}", "structure_sha256": digest})
        tasks.append(
            {
                "task_id": f"t{i}",
                "object_id": f"c{i}",
                "model_version": "m",
                "input_structure_sha256": digest,
                "structure_path": str(p),
                "stage": "dft_single_point",
                "status": "running" if i == 50 else "completed",
            }
        )
        if i < 50:
            records.append(
                {
                    "task_id": f"t{i}",
                    "status": "completed",
                    "converged": True,
                    "checks_passed": True,
                    "final_frame_valid": True,
                    "training_ready": True,
                }
            )
    state = {
        "tasks": tasks,
        "dft_dataset_records": records,
        "extrapolation_recovery_cycle": {
            "policy_version": "epoch_refresh_v1",
            "scope_id": "epoch_refresh",
            "invalid_model_version": "m",
            "round_evidence_digest": "bound",
            "direct_source_digest": "attempt",
            "selected": selected,
        },
    }
    facts = recovery_label_facts(state)
    assert facts["labelled_count"] == 50 and facts["threshold_met"] is True
    assert facts["status"] == "waiting" and facts["all_returned"] is False
    from phase_agent.policies.workflow_policy import action_errors

    state["extrapolation_recovery_cycle"]["status"] = "awaiting_reviewed_dft_labels"
    assert "extrapolation_recovery_dft_labels_required" in action_errors(
        {"tool": "update_mlip"}, state, {}
    )
    tasks[-1]["status"] = "failed"
    facts = recovery_label_facts(state)
    assert (
        facts["status"] == "complete"
        and facts["all_returned"] is True
        and facts["failed_count"] == 1
    )
    tasks[-1]["status"] = "cancelled"
    assert recovery_label_facts(state)["status"] == "waiting"


def test_forced_perturbation_follows_actual_activation_history():
    from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy

    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"initial": 1, "unrelated": 2, "m1": 3, "m2": 4}},
        "model_activation_history": [
            {"old_model_version": "initial", "new_model_version": "m1"},
            {"old_model_version": "m1", "new_model_version": "m2"},
        ],
    }
    facts = {"scope_id": "epoch_refresh", "evidence_digest": "bound", "failure_fraction": 0.1}
    assert epoch_recovery_policy(state, facts)["route"] == "perturbation"
    state["active_model_version"] = "m2"
    assert epoch_recovery_policy(state, facts)["route"] == "perturbation"


def test_ambiguous_activation_history_does_not_guess_sampling_route():
    from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy

    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2}},
        "model_activation_history": [
            {"old_model_version": "m0", "new_model_version": "m1"},
            {"old_model_version": "other", "new_model_version": "m1"},
        ],
    }
    facts = {"scope_id": "epoch_refresh", "evidence_digest": "bound", "failure_fraction": 0.1}
    assert epoch_recovery_policy(state, facts)["route"] == "perturbation"
