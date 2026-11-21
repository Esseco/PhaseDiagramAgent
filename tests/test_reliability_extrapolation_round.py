import pytest
from phase_agent.agents.reliability.extrapolation_round import assess_recovered_round


def task(i, failure=False):
    return {
        "task_id": str(i),
        "structure_id": str(i),
        "model_version": "m",
        "stage": "relax_and_feature",
        "status": "completed",
        "outputs": {
            "relaxation_observation": {
                "structure_id": str(i),
                "identity_scheme": "periodic_input_v1",
                "monitor_complete": True,
                "extrapolation_failure": failure,
            },
            "extrapolation_monitor": {
                "final_feature_difference": 1.5 if failure else 0.2,
                "extrapolation_failure": failure,
                "failures": [{"score": 1.5}] if failure else [],
            },
        },
    }


@pytest.mark.parametrize("kind", ["epoch_refresh", "relax_mc"])
def test_ten_percent_and_complete_negative(kind):
    data = [task(i, i == 0) for i in range(10)]
    result = assess_recovered_round(data, model_version="m", scope_complete=True, round_kind=kind)
    assert result["failure_fraction"] == 0.1 and result["direct_sampling_required"] is True
    data.append(task(10))
    assert (
        assess_recovered_round(data, model_version="m", scope_complete=True, round_kind=kind)[
            "direct_sampling_required"
        ]
        is False
    )


def test_missing_monitor_mc_scope_and_bad_model_are_not_safe():
    data = [task(0, True), task(1)]
    del data[1]["outputs"]["extrapolation_monitor"]
    assert (
        assess_recovered_round(
            data, model_version="m", scope_complete=True, round_kind="epoch_refresh"
        )["direct_sampling_required"]
        is None
    )
    mc = {
        "task_id": "mc",
        "model_version": "m",
        "stage": "deep_search",
        "outputs": {"extrapolation_failures": [{}]},
    }
    assert (
        assess_recovered_round([mc], model_version="m", scope_complete=True, round_kind="relax_mc")[
            "direct_sampling_required"
        ]
        is None
    )
    with pytest.raises(ValueError, match="model_mismatch"):
        assess_recovered_round(
            data, model_version="other", scope_complete=True, round_kind="epoch_refresh"
        )


def test_single_point_not_relaxed_is_not_denominator():
    data = [task(i, i == 0) for i in range(10)]
    data.append(
        {
            "task_id": "predict",
            "status": "completed",
            "model_version": "m",
            "stage": "relax_and_feature",
            "outputs": {
                "single_point_completed": True,
                "refresh_force_followup": {"relaxed": False},
            },
        }
    )
    result = assess_recovered_round(
        data, model_version="m", scope_complete=True, round_kind="epoch_refresh"
    )
    assert result["structure_count"] == 10 and result["direct_sampling_required"] is True


def test_reliability_gets_both_rounds_and_missing_scope_stays_unknown():
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds

    rows = [task(i, i == 0) for i in range(10)]
    state = {
        "active_model_version": "m",
        "tasks": rows,
        "model_refresh": {"new_model_version": "m", "task_ids": [str(i) for i in range(10)]},
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
    rounds = assess_current_rounds(state)["rounds"]
    assert rounds[0]["direct_sampling_required"] is True
    assert rounds[1]["direct_sampling_required"] is None
    state["model_refresh"]["task_ids"].append("missing")
    assert assess_current_rounds(state)["rounds"][0]["direct_sampling_required"] is None


def test_prompt_and_bounded_context_include_reliability_rule():
    from phase_agent.agents.reliability.prompts import INSTRUCTION
    from phase_agent.evidence.requests import summary_payload

    assert "10 percent" in INSTRUCTION and "Relax-MC" in INSTRUCTION
    facts = {
        "model_version": "m",
        "rounds": [{"round_kind": "epoch_refresh", "direct_sampling_required": True}],
    }
    result, _ = summary_payload(
        {"state": {}, "decision_context": {"reliability_extrapolation_rounds": facts}}
    )
    assert result["decision_context"]["reliability_extrapolation_rounds"] == facts


def test_completed_mc_requires_one_observation_per_step():
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
        "outputs": {
            "relaxation_observations": rows,
            "relaxation_observations_complete": True,
            "actual_mc_steps": 9,
        },
    }
    got = assess_recovered_round(
        [task], model_version="m", scope_complete=True, round_kind="relax_mc"
    )
    assert got["direct_sampling_required"] is True and got["failure_fraction"] == 0.1
    task["outputs"]["actual_mc_steps"] = 10
    assert (
        assess_recovered_round(
            [task], model_version="m", scope_complete=True, round_kind="relax_mc"
        )["direct_sampling_required"]
        is None
    )


def test_direct_gate_is_bound_to_round_and_blocks_ordinary_dft(tmp_path):
    import hashlib
    import json
    from phase_agent.agents.reliability.extrapolation_round import required_direct_round
    from phase_agent.policies.workflow_policy import action_errors

    tasks = [task(i, i == 0) for i in range(10)]
    state = {
        "active_model_version": "m",
        "tasks": tasks,
        "model_refresh": {"new_model_version": "m", "task_ids": [str(i) for i in range(10)]},
    }
    required = required_direct_round(state)
    assert required["scope_id"] == "epoch_refresh"
    errors = action_errors(
        {"tool": "select_dft_candidates", "parameters": {"decisions": []}}, state, {}
    )
    assert "reliability_direct_sampling_required:epoch_refresh" in errors
    assert not action_errors(
        {"tool": "prepare_local_batch_files", "parameters": {"mode": "direct_extrapolation"}},
        state,
        {},
    )
    plan = tmp_path / "selection.md"
    plan.write_text("test", encoding="utf8")
    manifest = {
        "model_version": "m",
        "round_evidence_digest": required["evidence_digest"],
        "feature_difference_filter": {"maximum": 1.0},
        "plan_path": str(plan),
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    }
    result_file = tmp_path / "direct_result.json"
    result_file.write_text(json.dumps(manifest), encoding="utf8")
    state["direct_sampling_results"] = {
        "epoch_refresh": {
            **manifest,
            "manifest_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
        }
    }
    assert required_direct_round(state) is None
    plan.write_text("changed", encoding="utf8")
    assert required_direct_round(state) is not None
    plan.write_text("test", encoding="utf8")
    tasks[1]["outputs"]["extrapolation_monitor"]["final_feature_difference"] = 0.3
    assert required_direct_round(state) is not None


def test_refresh_gate_yields_to_mandatory_direct():
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    tasks = [task(i, i == 0) for i in range(10)]
    state = {
        "active_model_version": "m",
        "tasks": tasks,
        "model_refresh": {
            "status": "waiting_results",
            "new_model_version": "m",
            "task_ids": [str(i) for i in range(10)],
        },
    }
    assert run_model_refresh_gate(state, {}, registry={}, context={}) is None


@pytest.mark.parametrize("kind", ["epoch_refresh"])
def test_trigger_proposes_direct_through_existing_action_graph(kind):
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    tasks = [task(i, i == 0) for i in range(10)]
    if kind == "epoch_refresh":
        state = {
            "active_model_version": "m",
            "tasks": tasks,
            "model_refresh": {"new_model_version": "m", "task_ids": [str(i) for i in range(10)]},
        }
    else:
        observations = [
            {
                "structure_id": str(i),
                "identity_scheme": "periodic_input_v1",
                "monitor_complete": True,
                "extrapolation_failure": i == 0,
            }
            for i in range(10)
        ]
        state = {
            "active_model_version": "m",
            "tasks": [
                {
                    "task_id": "mc",
                    "model_version": "m",
                    "stage": "deep_search",
                    "status": "completed",
                    "relax_mc_bundle_id": "b",
                    "outputs": {
                        "relaxation_observations": observations,
                        "relaxation_observations_complete": True,
                        "actual_mc_steps": 9,
                    },
                }
            ],
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
    state["upload_layout"] = {"model_rounds": {"m": 3}}
    state["pending_execution_policies"] = {}
    called = []
    result = select_initial_proposal(
        current=state,
        mode="interactive",
        config={"agent": {"allowed_tools": ["prepare_local_batch_files"]}},
        context={},
        decision_state={},
        registry={"prepare_local_batch_files": {"handler": lambda **kw: None}},
        agent_client=lambda payload: None,
        invocation_id="test",
        propose=lambda decision, **kwargs: {
            "operation": "sample_direct_extrapolation", "tool": "prepare_local_batch_files",
            "parameters": {"mode": "direct_extrapolation", "target": 100,
                           "scope_id": decision["decision_context"]["mandatory_direct_round"]["scope_id"],
                           "round_evidence_digest": decision["decision_context"]["mandatory_direct_round"]["evidence_digest"]},
            "task_key": "direct-test", "budget": 0.0,
            "reason": "Strategic accepted", "expected_purpose": "DIRECT review"},
        revise=None,
        prepare_debug=None,
        is_verified_second=None,
        mc_block=None,
        model_failed=None,
        record_id_factory=lambda current, invocation: "record",
    )
    action = result["proposal"]["raw_action"]
    assert (
        action["operation"] == "sample_direct_extrapolation"
        and action["parameters"]["target"] == 100
    )
    assert action["parameters"]["scope_id"] == ("epoch_refresh" if kind == "epoch_refresh" else "b")
    assert result["_proposal_selected"] and not called


def test_mandatory_trigger_consults_reliability_even_when_plan_omits_it(monkeypatch):
    from types import SimpleNamespace
    from phase_agent.graphs.decision_team.nodes import consultation_plan
    import phase_agent.agents.strategic.agent as strategic

    monkeypatch.setattr(
        strategic,
        "plan_consultation",
        lambda client, snapshot, proposal: {
            "consult": [],
            "question": "assess",
            "snapshot_id": "snap",
        },
    )
    snapshot = {
        "facts": {
            "decision_context": {
                "reliability_extrapolation_rounds": {"rounds": [{"direct_sampling_required": True}]}
            }
        }
    }
    result = consultation_plan(
        {"snapshot": snapshot}, SimpleNamespace(context=SimpleNamespace(expert_call=None))
    )
    assert result["consultation"]["consult"] == ["reliability"]


def test_mandatory_direct_waits_when_strategic_does_not_agree(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    monkeypatch.setattr(extrapolation_round, "required_direct_round", lambda state: {
        "scope_id": "epoch_refresh", "evidence_digest": "bound"})
    result = select_initial_proposal(
        current={}, mode="interactive", config={"agent": {"allowed_tools": ["prepare_local_batch_files"]}},
        context={}, decision_state={}, registry={"prepare_local_batch_files": {"handler": lambda **kw: None}},
        agent_client=lambda payload: None, invocation_id="test",
        propose=lambda *args, **kwargs: {"tool": "pause_search"}, revise=None,
        prepare_debug=None, is_verified_second=None, mc_block=None, model_failed=None,
        record_id_factory=lambda *args: "record")
    assert result["status"] == "not_configured" and result["submitted"] is False


def test_explicit_direct_proposal_passes_refresh_gate_without_faking_completion():
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    state = {
        "active_model_version": "m",
        "tasks": [task(0)],
        "model_refresh": {"status": "waiting_results", "new_model_version": "m", "task_ids": ["0"]},
    }
    assert (
        run_model_refresh_gate(
            state, {}, registry={}, context={"explicit_direct_scope": "epoch_refresh"}
        )
        is None
    )
    assert state["model_refresh"]["status"] == "waiting_results"


def test_completed_direct_requires_quarantine_before_followup(monkeypatch):
    from phase_agent.agents.reliability.extrapolation_round import unquarantined_direct_round
    from phase_agent.policies.workflow_policy import action_errors
    from phase_agent.tools.workflows import direct_extrapolation_sampling
    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda receipt: True)
    state = {"active_model_version": "bad", "direct_sampling_results": {"scope": {"model_version": "bad", "source_digest": "digest"}}}
    assert unquarantined_direct_round(state) == "scope"
    assert "direct_recovery_quarantine_required:scope" in action_errors({"tool": "select_dft_candidates", "parameters": {"decisions": []}}, state, {})
    state["extrapolation_recovery_cycle"] = {"scope_id": "scope", "invalid_model_version": "bad", "direct_source_digest": "digest"}
    assert unquarantined_direct_round(state) is None


def test_strategic_can_propose_perturbation_route_bound_to_round(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    monkeypatch.setattr(extrapolation_round, 'required_direct_round', lambda state: {'scope_id':'epoch_refresh', 'evidence_digest':'bound', 'failure_fraction':0.1})
    action = {'operation':'prepare_perturbation_recovery', 'tool':'prepare_local_batch_files',
              'parameters':{'mode':'perturbation_recovery_inputs', 'scope_id':'epoch_refresh',
                            'round_evidence_digest':'bound', 'max_det_H':2, 'parent_record_ids':['p']},
              'task_key':'p', 'budget':0, 'reason':'coverage'}
    result = select_initial_proposal(current={'active_model_version':'m', 'upload_layout':{'model_rounds':{'m':1}}}, mode='interactive', config={'agent': {'allowed_tools':['prepare_local_batch_files']}},
        context={}, decision_state={}, registry={'prepare_local_batch_files': {'handler':lambda **kwargs: None}},
        agent_client=lambda payload:None, invocation_id='i', propose=lambda *a, **k: action, revise=None,
        prepare_debug=None, is_verified_second=None, mc_block=None, model_failed=None,
        record_id_factory=lambda *a:'record')
    assert result['_proposal_selected'] and result['proposal']['raw_action']['operation'] == 'prepare_perturbation_recovery'


def test_perturbation_waits_for_all_mh1_results(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    monkeypatch.setattr(extrapolation_round, 'required_direct_round', lambda state: {'scope_id':'epoch_refresh', 'evidence_digest':'bound'})
    state = {'perturbation_recovery_batch': {'scope_id':'epoch_refresh', 'round_evidence_digest':'bound', 'task_ids':['one']},
             'tasks':[{'task_id':'one','status':'running'}]}
    result = select_initial_proposal(current=state, mode='interactive', config={'agent': {'allowed_tools':['prepare_local_batch_files']}},
        context={}, decision_state={}, registry={'prepare_local_batch_files': {'handler':lambda **kwargs: None}},
        agent_client=lambda payload:None, invocation_id='i', propose=lambda *a, **k: pytest.fail('must not propose while running'),
        revise=None, prepare_debug=None, is_verified_second=None, mc_block=None, model_failed=None,
        record_id_factory=lambda *a:'record')
    assert result['status'] == 'waiting_results' and result['state']['tasks'][0]['status'] == 'running'


def test_switch_to_perturbation_requires_all_old_dft_tasks_terminal(tmp_path):
    import hashlib
    from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
    paths = [tmp_path / 'a', tmp_path / 'b']
    for path in paths:
        path.write_text(path.name)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    cycle = {'selected':[{'structure_sha256':sha(p)} for p in paths]}
    state = {'tasks':[{'stage':'dft_single_point','status':'failed','structure_path':str(paths[0])},
                      {'stage':'dft_single_point','status':'running','structure_path':str(paths[1])}]}
    assert not recovery_dft_failure_settled(state, cycle)
    state['tasks'][1]['status'] = 'completed'
    assert recovery_dft_failure_settled(state, cycle)
    state['tasks'][0]['status'] = 'completed'
    assert not recovery_dft_failure_settled(state, cycle)


def test_successful_retry_prevents_unnecessary_route_switch(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
    from phase_agent.science.dft import spin_acceptance
    monkeypatch.setattr(spin_acceptance, 'spin_standard_passed', lambda record: True)
    frame = tmp_path / 'frame'
    frame.write_text('frame')
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    cycle = {'selected':[{'structure_sha256':digest}]}
    state = {'extrapolation_recovery_cycle':cycle, 'tasks':[
        {'task_id':'failed','stage':'dft_single_point','status':'failed','structure_path':str(frame)},
        {'task_id':'success','stage':'dft_single_point','status':'completed','structure_path':str(frame)}],
        'dft_dataset_records':[{'task_id':'success','status':'completed','converged':True,
             'checks_passed':True,'final_frame_valid':True,'training_ready':True}]}
    assert not recovery_dft_failure_settled(state, cycle)


def test_verified_historical_endpoint_lower_bound_never_claims_step_monitoring(tmp_path, monkeypatch):
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds
    import phase_agent.evidence.extrapolation_review as review
    rows = [task(i) for i in range(10)]
    for row in rows:
        row['outputs'] = {}
    rows[0]['extrapolation_review'] = {'status': 'extrapolation_failure'}
    (tmp_path / 'evidence.csv').write_text('task_id,classification\n1,extrapolation_failure\n', encoding='utf8')
    state = {'active_model_version':'m', 'tasks':rows,
             'model_refresh':{'new_model_version':'m','task_ids':[str(i) for i in range(10)]},
             'endpoint_supplement':{'directory':str(tmp_path)}}
    monkeypatch.setattr(review, 'extrapolation_review', lambda state: {
        'status':'endpoint_classified_incomplete','coverage_complete':True,'task_count':10})
    result = assess_current_rounds(state)['rounds'][0]
    assert result['direct_sampling_required'] is True
    assert result['endpoint_failure_structure_count'] == 2
    assert result['status'] == 'endpoint_failure_lower_bound'
    assert result['scope_complete'] is False and result['force_jump_evidence_available'] is False
    monkeypatch.setattr(review, 'extrapolation_review', lambda state: {'status':'unverified'})
    assert assess_current_rounds(state)['rounds'][0]['direct_sampling_required'] is None
