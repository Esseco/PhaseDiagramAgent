from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
from ase import Atoms
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from phase_agent.tools.workflows.direct_extrapolation_sampling import prepare_direct_sampling


def make_case(tmp_path, monkeypatch):
    module = tmp_path / "maml"
    model = module / "describers/data/m3gnet_models/matbench_mp_e_form/0/m3gnet"
    model.mkdir(parents=True)
    (model / "weight").write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "maml", SimpleNamespace(__file__=str(module / "__init__.py")))
    digest = hashlib.sha256(b"weight\0weights").hexdigest()
    from Process_MLIP.direct_sampling import features, workflow

    monkeypatch.setattr(
        features, "M3GNetEmbedding", lambda path: SimpleNamespace(metadata={"model_sha256": digest})
    )

    def extract(paths, encoder, cache, *, progress=None):
        with h5py.File(cache, "w") as h:
            h.attrs["sources"] = json.dumps([str(x) for x in paths])

    def run(cache, out, target, reference_count=0):
        with h5py.File(cache, "r") as h:
            paths = json.loads(h.attrs["sources"])
        out = Path(out)
        out.mkdir()
        (out / "selected_indices.json").write_text(
            json.dumps([{"source": paths[reference_count], "frame": 0}])
        )
        return {"n_samples": 1}

    monkeypatch.setattr(workflow, "extract", extract)
    monkeypatch.setattr(workflow, "run", run)
    atoms = Atoms(
        "NaCoO2", positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True
    )
    frame = tmp_path / "failure.vasp"
    write(frame, atoms, format="vasp")
    checksum = hashlib.sha256(frame.read_bytes()).hexdigest()
    cand = {
        "candidate_id": "EXTRAP-t-0",
        "structure_id": "s",
        "source_task_id": "t",
        "structure_path": str(frame),
        "structure_sha256": checksum,
        "model_version": "m",
        "extrapolation_failure": True,
    }
    reference = {
        "task_id": "d",
        "status": "completed",
        "checks_passed": True,
        "converged": True,
        "final_frame_valid": True,
        "training_ready": True,
        "structure": AseAtomsAdaptor.get_structure(atoms).as_dict(),
    }
    state = {
        "active_model_version": "m",
        "model_refresh": {"new_model_version": "m", "task_ids": ["t"]},
        "tasks": [
            {
                "task_id": "t",
                "structure_id": "s",
                "model_version": "m",
                "outputs": {
                    "extrapolation_failures": [
                        {
                            "structure_path": str(frame),
                            "structure_checksum": checksum,
                            "extrapolation_monitor": {"extrapolation_failure": True, "final_feature_difference": 0.5},
                        }
                    ]
                },
            }
        ],
        "qbc_candidates": [cand],
        "dft_dataset_records": [reference],
    }
    context = {"event_state": state, "effective_config": {"phase_diagram_directory": str(tmp_path)}}
    action = {"parameters": {"scope_id": "epoch_refresh", "target": 100}}
    return state, context, action, frame


def test_direct_uses_exact_failure_frames_and_dft_reference(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    result = prepare_direct_sampling(action=action, context=context)
    assert result["status"] == "completed" and not result["submitted"]
    assert (
        result["direct_sampling"]["reference_count"] == 1
        and result["direct_sampling"]["selected_count"] == 1
    )


def test_direct_review_uses_verified_parent_phase_and_frame(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.direct_review_report import publish_direct_review, bind_direct_candidates

    state, context, action, frame = make_case(tmp_path, monkeypatch)
    state["tasks"][0]["branch_id"] = "b"
    result = prepare_direct_sampling(action=action, context=context)
    saved = result["state"]
    ledger = {"branches": {"b": {"P": "P3", "H": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}}}
    report = publish_direct_review(saved, ledger, "epoch_refresh")
    assert report["parent_phase_counts"] == {"P3": 1}
    assert "| P3 | 2 |" in Path(report["path"]).read_text(encoding="utf8")
    bound, binding = bind_direct_candidates(saved, ledger, "epoch_refresh")
    assert binding["selected_count"] == 1
    assert bound["qbc_candidates"][0]["branch_id"] == "b"
    assert bound["qbc_candidates"][0]["phase_basis"] == "parent_branch"
    frame.write_text("changed")
    with pytest.raises(ValueError, match="selected_frame_invalid"):
        publish_direct_review(saved, ledger, "epoch_refresh")


def test_direct_samples_complete_batch_not_only_failures(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    other = tmp_path / "normal.vasp"
    atoms = Atoms(
        "NaCoO2", positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[6, 6, 6], pbc=True
    )
    write(other, atoms, format="vasp")
    state["tasks"].append(
        {
            "task_id": "t2",
            "structure_id": "s2",
            "model_version": "m",
            "status": "completed",
            "outputs": {
                "structure_path": str(other),
                "structure_checksum": hashlib.sha256(other.read_bytes()).hexdigest(),
                "extrapolation_monitor": {"final_feature_difference": 0.5},
            },
        }
    )
    state["model_refresh"]["task_ids"].append("t2")
    result = prepare_direct_sampling(action=action, context=context)
    assert result["direct_sampling"]["batch_task_count"] == 2
    assert result["direct_sampling"]["unique_candidate_count"] == 2
    assert "本批任务 2" in Path(result["direct_sampling"]["plan_path"]).read_text(encoding="utf8")
    assert (
        result["direct_sampling"]["selected"][0]["structure_sha256"]
        == hashlib.sha256(frame.read_bytes()).hexdigest()
    )
    assert (
        Path(result["direct_sampling"]["plan_path"]).is_file()
        and state.get("direct_sampling_results") is None
    )
    assert (
        prepare_direct_sampling(action=action, context=context)["direct_sampling"]["source_digest"]
        == result["direct_sampling"]["source_digest"]
    )


def test_owned_state_keeps_input_unchanged_and_records_stage_timings(tmp_path, monkeypatch):
    state, context, action, _ = make_case(tmp_path, monkeypatch)
    state["analysis_step_diagnostics"] = {"direct": "previous"}
    context["_event_state_owned"] = True
    result = prepare_direct_sampling(action=action, context=context)
    assert "direct_sampling_results" not in state
    assert state["analysis_step_diagnostics"] == {"direct": "previous"}
    assert result["state"]["tasks"] is state["tasks"]  # shared, read-only large input
    assert result["state"]["qbc_candidates"] is not state["qbc_candidates"]
    assert all(value >= 0 for value in result["direct_sampling"]["timing_seconds"].values())


def test_direct_rejects_changed_frame_and_wrong_round(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    frame.write_text("changed")
    with pytest.raises(ValueError, match="failure_candidate_missing"):
        prepare_direct_sampling(action=action, context=context)
    action["parameters"]["scope_id"] = "other"
    with pytest.raises(ValueError, match="batch_not_verified"):
        prepare_direct_sampling(action=action, context=context)


def test_direct_sampling_uses_epoch_or_latest_unassigned(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    result = prepare_direct_sampling(action=action, context=context)
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    assert Path(result["direct_sampling"]["plan_path"]).is_relative_to(
        analysis_metadata_directory(tmp_path) / "Unassigned"
    )
    assert not (tmp_path / "model_epoch_unrecorded").exists()
    state["upload_layout"] = {"model_rounds": {"m": 2}}
    result = prepare_direct_sampling(action=action, context=context)
    assert Path(result["direct_sampling"]["plan_path"]).is_relative_to(
        tmp_path / "epoch1" / "direct_sampling"
    )


def test_direct_sampling_reuses_existing_unassigned_legacy_receipt(tmp_path, monkeypatch):
    import shutil

    state, context, action, _ = make_case(tmp_path, monkeypatch)
    first = prepare_direct_sampling(action=action, context=context)
    saved = Path(first["direct_sampling"]["plan_path"]).parent
    legacy = tmp_path / "model_epoch_unrecorded" / "direct_sampling" / saved.name
    legacy.parent.mkdir(parents=True)
    shutil.move(str(saved), legacy)
    # Legacy manifests retain their original paths; use an existing valid receipt
    # only when its persisted references still match the copied source.
    manifest_path = legacy / "direct_result.json"
    data = json.loads(manifest_path.read_text(encoding="utf8"))
    data["plan_path"] = str(legacy / "selection.md")
    manifest_path.write_text(json.dumps(data), encoding="utf8")
    second = prepare_direct_sampling(action=action, context=context)
    assert Path(second["direct_sampling"]["plan_path"]).parent == legacy
    assert second["direct_sampling"]["source_digest"] == first["direct_sampling"]["source_digest"]


def test_whole_batch_includes_completed_and_below_threshold_archive(tmp_path, monkeypatch):
    import csv
    import zipfile
    from phase_agent.tools.workflows.direct_extrapolation_sampling import _whole_batch_candidates
    from phase_agent.analysis.phase import phase_dft_candidates

    monkeypatch.setattr(phase_dft_candidates, "extrapolation_dft_candidates", lambda *a: [])
    from phase_agent.evidence import endpoint_supplement
    monkeypatch.setattr(endpoint_supplement, "supplement_facts", lambda state: {"task_ids": ["t2"]})
    normal = tmp_path / "normal.vasp"
    normal.write_bytes(b"normal frame")
    payload = b"unconverged below threshold frame"
    directory = tmp_path / "evidence"
    directory.mkdir()
    with (directory / "evidence.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task_id", "archive_path", "final_sha256", "classification"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "t2",
                "archive_path": "batch/t2/initial_relaxed.vasp",
                "final_sha256": hashlib.sha256(payload).hexdigest(),
                "classification": "below_threshold",
            }
        )
    with zipfile.ZipFile(directory / "frames.zip", "w") as archive:
        archive.writestr("batch/t2/initial_relaxed.vasp", payload)
    tasks = {
        "t1": {
            "task_id": "t1",
            "structure_id": "s1",
            "status": "completed",
            "outputs": {
                "structure_path": str(normal),
                "structure_checksum": hashlib.sha256(normal.read_bytes()).hexdigest(),
            },
        },
        "t2": {"task_id": "t2", "structure_id": "s2", "status": "failed", "outputs": {}},
    }
    rows, materialize = _whole_batch_candidates(
        {"endpoint_supplement": {"directory": str(directory)}}, tasks, set(tasks), "m"
    )
    assert [r["source_task_id"] for r in rows] == ["t1", "t2"]
    assert materialize == {"t2": payload}
    assert all(r["extrapolation_failure"] is False for r in rows)


def test_mc_direct_requires_each_relaxed_frame_and_checks_checksum(tmp_path):
    from phase_agent.tools.workflows.direct_extrapolation_sampling import _mc_batch_candidates

    observations = []
    for index in range(2):
        path = tmp_path / f"frame{index}.vasp"
        path.write_bytes(f"frame{index}".encode())
        observations.append({"monitor_complete": True, "extrapolation_failure": index == 1,
                             "structure_path": str(path), "structure_checksum": hashlib.sha256(path.read_bytes()).hexdigest()})
    tasks = {"mc": {"structure_id": "s", "branch_id": "b", "stage": "deep_search", "outputs": {
        "actual_mc_steps": 1, "relaxation_observations_complete": True, "relaxation_observations": observations}}}
    rows, extra = _mc_batch_candidates(tasks, {"mc"}, "m")
    assert len(rows) == 2 and extra == {}
    assert [row["extrapolation_failure"] for row in rows] == [False, True]
    del observations[0]["structure_path"]
    with pytest.raises(ValueError, match="complete_frame_inventory_required"):
        _mc_batch_candidates(tasks, {"mc"}, "m")


def test_direct_evidence_preparation_does_not_require_dft_vs_search_budget_comparison():
    from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors

    context = {"round_budget_evidence": {"required": True}}
    assert not round_budget_review_errors({"operation": "sample_direct_extrapolation"}, context)
    assert round_budget_review_errors({"tool": "select_dft_candidates"}, context)


def test_direct_scope_survives_agent_client_wrapper():
    from phase_agent.models.capabilities import copy_model_capabilities

    source = SimpleNamespace(_codex_structured=True, _explicit_direct_scope="epoch_refresh")
    target = SimpleNamespace()
    copy_model_capabilities(source, target)
    assert target._explicit_direct_scope == "epoch_refresh"


def test_refresh_gate_allows_direct_evidence_but_not_ordinary_search():
    from phase_agent.policies.branch_conditions import branch_entry_errors

    state = {"model_refresh": {"status": "waiting_results"}}
    direct = {"tool": "prepare_local_batch_files", "operation": "sample_direct_extrapolation",
              "parameters": {"mode": "direct_extrapolation"}}
    assert "new_model_structure_refresh_required" not in branch_entry_errors(direct, state)
    assert "new_model_structure_refresh_required" in branch_entry_errors({"tool": "generate_branches"}, state)


def test_large_failure_does_not_publish_or_replace_latest_hull(tmp_path, monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    monkeypatch.setattr(extrapolation_round, "assess_current_rounds", lambda state: {
        "rounds": [{"scope_id": "epoch_refresh", "direct_sampling_required": True}]})
    latest = tmp_path / "latest.txt"
    latest.write_text("previous valid diagram", encoding="utf8")
    result = update_phase_diagram([], active_model_version="m", output_state={"active_model_version": "m"},
                                  output_directory=tmp_path)
    assert result["diagrams"]["mlip"]["status"] == "invalidated"
    assert result["diagrams"]["mlip"]["invalidated_rounds"] == ["epoch_refresh"]
    assert latest.read_text(encoding="utf8") == "previous valid diagram"
    assert list(tmp_path.iterdir()) == [latest]


def test_mc_direct_batch_includes_initial_relax_and_all_mc_frames(tmp_path):
    from phase_agent.tools.workflows.direct_extrapolation_sampling import _mc_batch_candidates
    initial = tmp_path / "initial.vasp"
    initial.write_bytes(b"initial")
    mc = tmp_path / "mc.vasp"
    mc.write_bytes(b"mc")
    checksum = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    tasks = {"relax": {"task_id": "relax", "structure_id": "s0", "branch_id": "b", "stage": "relax_and_feature", "status": "completed", "model_version": "m", "outputs": {"structure_path": str(initial), "structure_checksum": checksum(initial)}}, "mc": {"task_id": "mc", "structure_id": "s1", "branch_id": "b", "stage": "deep_search", "model_version": "m", "outputs": {"actual_mc_steps": 0, "relaxation_observations_complete": True, "relaxation_observations": [{"monitor_complete": True, "extrapolation_failure": False, "structure_path": str(mc), "structure_checksum": checksum(mc)}]}}}
    rows, _ = _mc_batch_candidates(tasks, set(tasks), "m")
    assert {row["source_task_id"] for row in rows} == {"relax", "mc"}


def test_direct_excludes_greater_than_one_and_unknown_but_keeps_one(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    state['tasks'][0]['outputs']['extrapolation_failures'][0]['extrapolation_monitor']['final_feature_difference'] = 1.01
    for tid, score in [('t2', 1.0), ('t3', None)]:
        target = tmp_path / (tid + '.vasp')
        write(target, Atoms('NaCoO2', positions=[[0,0,0],[1,1,1],[1,0,0],[0,1,0]], cell=[6,6,6], pbc=True), format='vasp')
        task = {'task_id': tid, 'structure_id': tid, 'model_version': 'm', 'status': 'completed',
                'outputs': {'structure_path': str(target), 'structure_checksum': hashlib.sha256(target.read_bytes()).hexdigest(),
                            'extrapolation_monitor': {'final_feature_difference': score}}}
        state['tasks'].append(task)
        state['model_refresh']['task_ids'].append(tid)
    receipt = prepare_direct_sampling(action=action, context=context)['direct_sampling']
    assert receipt['feature_difference_filter'] == {'maximum': 1.0, 'excluded': {'above_threshold': 1, 'unknown': 1}}
    assert [r['source_task_id'] for r in receipt['selected']] == ['t2']


def test_direct_rejects_when_feature_difference_unavailable(tmp_path, monkeypatch):
    state, context, action, _ = make_case(tmp_path, monkeypatch)
    del state['tasks'][0]['outputs']['extrapolation_failures'][0]['extrapolation_monitor']['final_feature_difference']
    with pytest.raises(ValueError, match='verified_feature_difference'):
        prepare_direct_sampling(action=action, context=context)


def test_shared_direct_accepts_verified_mh1_perturbation_and_dft_reference(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    old_hull = {'status': 'completed', 'version': 'old-hull', 'model_version': 'old', 'entries': []}
    state['phase_diagrams'] = {'mlip': {'status': 'completed', 'version': 'bad-hull', 'model_version': 'm', 'entries': []}}
    state['model_refresh'].update(old_diagram=old_hull, old_model_version='old')
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds
    round_digest = assess_current_rounds(state)['rounds'][0]['evidence_digest']
    input_frame = tmp_path / 'perturb-input.vasp'
    input_frame.write_bytes(frame.read_bytes())
    digest = hashlib.sha256(input_frame.read_bytes()).hexdigest()
    model = {'version': 'mace-mh-1', 'sha256': 'a' * 64}
    manifest = {'schema': 'perturbation-recovery-inputs-v1', 'scope_id': 'epoch_refresh',
                'round_evidence_digest': round_digest, 'failed_model_version': 'm', 'relax_model': model,
                'frames': [{'sha256': digest, 'path': input_frame.name, 'branch_id': 'b'}],
                'parameters': {'min_distance_A': 1.0, 'volume_ratio': [0.75, 1.25]}}
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest), encoding='utf8')
    task = {'task_id': 'mh1', 'structure_id': 'perturbed', 'branch_id': 'b', 'model_version': 'mace-mh-1',
            'status': 'completed', 'input_structure_sha256': digest, 'perturbation_batch_id': 'batch', 'model_sha256': 'a' * 64,
            'outputs': {'structure_path': str(frame), 'structure_checksum': digest, 'verified_model_sha256': 'a' * 64,
                        'extrapolation_monitor': {'extrapolation_failure': False, 'failures': [], 'final_feature_difference': 0.2}}}
    state['tasks'].append(task)
    state['perturbation_recovery_batch'] = {'status': 'results_received', 'scope_id': 'epoch_refresh',
        'round_evidence_digest': round_digest, 'approval_record_id': 'approved', 'batch_id': 'batch',
        'manifest_path': str(mp), 'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(), 'task_ids': ['mh1']}
    action['parameters']['recovery_route'] = 'perturbation'
    result = prepare_direct_sampling(action=action, context=context)
    receipt = result['direct_sampling']
    assert receipt['recovery_route'] == 'perturbation' and receipt['reference_count'] == 1
    assert receipt['selected'][0]['source_task_id'] == 'mh1'
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh, recovery_label_facts, recovery_model_matches_labels
    from phase_agent.tools.workflows.validate_dft_agent_decisions import _is_direct_recovery_selection
    from phase_agent.science.dft import spin_acceptance
    quarantined, _ = quarantine_refresh(result['state'], 'epoch_refresh')
    assert quarantined['phase_diagrams']['mlip'] == old_hull
    assert _is_direct_recovery_selection(quarantined, {'PERTURB-mh1'})
    monkeypatch.setattr(spin_acceptance, 'spin_standard_passed', lambda record: True)
    quarantined['tasks'].append({'task_id': 'new-dft', 'stage': 'dft_single_point', 'status': 'completed', 'structure_path': str(frame)})
    quarantined['dft_dataset_records'].append({'task_id': 'new-dft', 'status': 'completed', 'converged': True,
        'checks_passed': True, 'training_ready': True, 'final_frame_valid': True})
    assert recovery_label_facts(quarantined)['status'] == 'complete'
    quarantined['remote_finetune_jobs'] = {'j': {'status': 'results_received', 'original_model_version': 'm',
        'training_handoff': {'candidate_model_version': 'm-next'}, 'report': {'dataset_scope': {'data_ids': ['new-dft:frame:0']}}}}
    assert recovery_model_matches_labels(quarantined, 'm-next')
    task['outputs']['structure_checksum'] = 'bad'
    with pytest.raises(ValueError, match='final_frame_invalid'):
        prepare_direct_sampling(action=action, context=context)


def test_synthetic_perturbation_pipeline_reaches_shared_training_scope(tmp_path, monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.tools.local.prepare_perturbation_relax import prepare_perturbation_relax
    from phase_agent.tools.workflows import perturbation_staging
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh, recovery_label_facts, recovery_model_matches_labels
    from phase_agent.science.dft import spin_acceptance

    state, direct_context, _, source = make_case(tmp_path, monkeypatch)
    h = [[2,0,0],[0,1,0],[0,0,1]]
    manager = PhaseDataManager({'P':['O3'], 'H':{'O3':[h]}, 'TM_ratio':{'Co':1}})
    bid = manager.add_branch(P='O3', H=h, x=0.5, T=['Co'], composition={'Na':1,'Co':1,'O':2})
    ledger = tmp_path / 'ledger.json'
    manager.save(ledger)
    old = {'status':'completed','version':'old-hull','model_version':'old','entries':[
        {'composition':{'O':2},'eform_per_O2':0,'phase':'O1','is_stable':True},
        {'composition':{'Na':1,'O':2},'eform_per_O2':0,'phase':'O3','is_stable':True}]}
    state['phase_diagrams'] = {'mlip': {'status':'completed','version':'bad-hull','model_version':'m','entries':[]}}
    state['model_refresh'].update(old_diagram=old, old_model_version='old')
    state['model_registry'] = {'mace-mh-1': {'model': {'framework':'mace', 'version':'mace-mh-1',
        'model_path':'/models/mh1.model', 'mace_head':'omat_pbe'}, 'comparison_model_sha256':'a'*64}}
    round_digest = extrapolation_round.assess_current_rounds(state)['rounds'][0]['evidence_digest']
    monkeypatch.setattr(perturbation_staging, 'required_direct_round', lambda current: {'scope_id':'epoch_refresh','evidence_digest':round_digest})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(perturbation_staging, 'build_perturbation_recovery_pool', lambda *a, **kw: {
        'hull_version':'old-hull', 'model_version':'old', 'candidates':[{'record_id':'parent','structure_id':'s',
        'branch_id':bid, 'structure_path':str(source), 'structure_sha256':digest, 'phase':'O3',
        'x_Na_per_O2':0.5,'H':h,'det_H':2}]})
    config = {'budgets':default_budget_rules(),'ledger_path':str(ledger), 'phase_diagram_directory':str(tmp_path),
        'upload_batches_directory':str(tmp_path/'uploads'), 'supercomputer':{'worker':{'command':['python','-m',
        'phase_agent.tools.remote.run_slurm_array_task','--executor','phase_agent.science.mlip.slurm_executor:execute_mlip_task']}}}
    params = {'scope_id':'epoch_refresh','round_evidence_digest':round_digest,'max_det_H':2,
              'parent_record_ids':['parent'],'frames_per_parent':2,'seed':42}
    state['dedup_gate'] = {'status':'ready', 'valid_structure_ids':['old-validated-structure']}
    config['supercomputer']['batch_sizes'] = {'relax_and_feature': 1}
    context = {'event_state':state,'manager':manager,'effective_config':config,'config_version':'cfg','approval_record_id':'approved'}
    prepared = prepare_perturbation_relax(action={'parameters':params}, context=context)
    assert prepared['state']['dedup_gate'] == state['dedup_gate']
    assert prepared['status'] == 'prepared' and Path(prepared['handoff']['upload_archive']).is_file()
    recovered = prepared['state']
    batch = recovered['perturbation_recovery_batch']
    for task in recovered['tasks']:
        if task.get('task_id') in batch['task_ids']:
            task['status'] = 'completed'
            task['outputs'] = {'structure_path': task['structure_path'], 'structure_checksum': task['input_structure_sha256'],
                'verified_model_sha256':'a'*64, 'extrapolation_monitor':{'extrapolation_failure':False,'failures':[], 'final_feature_difference':0.2}}
    batch['status'] = 'results_received'
    from phase_agent.tools.workflows.perturbation_direct import previously_approved_direct
    recovered.setdefault('pending_execution_policies', {})
    recovered['action_records'] = [{'record_id':'approved', 'status':'prepared', 'human_feedback':{'decision':'approve'},
        'final_action':{'operation':'prepare_perturbation_recovery','parameters':params}}]
    direct_action = {'operation':'sample_direct_extrapolation', 'tool':'prepare_local_batch_files',
        'parameters':{'mode':'direct_extrapolation', 'scope_id':'epoch_refresh',
                      'round_evidence_digest':round_digest, 'recovery_route':'perturbation'}}
    assert previously_approved_direct(recovered, direct_action)
    params['direct_target'] = 100
    direct_action['parameters']['target'] = 99
    assert not previously_approved_direct(recovered, direct_action)
    direct_action['parameters']['target'] = 100
    assert previously_approved_direct(recovered, direct_action)
    params.pop('direct_target')
    direct_action['parameters'].pop('target')
    direct_action['parameters']['round_evidence_digest'] = 'altered'
    assert not previously_approved_direct(recovered, direct_action)
    direct_action['parameters']['round_evidence_digest'] = round_digest
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    monkeypatch.setattr(extrapolation_round, 'required_direct_round', lambda current: {'scope_id':'epoch_refresh','evidence_digest':round_digest})
    planned = select_initial_proposal(current=recovered, mode='interactive', config={'agent':{'allowed_tools':['prepare_local_batch_files']}},
        context={'manager':manager}, decision_state={}, registry={'prepare_local_batch_files':{'handler':lambda **kw:None}},
        agent_client=lambda payload:None, invocation_id='direct', propose=lambda *a, **k:direct_action,
        revise=None, prepare_debug=None, is_verified_second=None, mc_block=None, model_failed=None,
        record_id_factory=lambda *a:'next')
    assert planned['_proposal_selected'] and planned['current']['perturbation_recovery_batch']['status'] == 'results_received'
    from phase_agent.tools.workflows.run_tool_step import _approval_graph_node
    frame = {'current':planned['current'], 'proposal':planned['proposal'], 'mode':'interactive', 'human_feedback':None,
             'replay_record':None, 'stored':None, 'record_id':'next', 'pending_key':'pending', 'invocation_id':'direct',
             'context':{}, 'config':{}}
    approved = _approval_graph_node(frame)
    assert approved['policy']['execute'] is True and approved['policy']['authorization_source'] == 'approved_extrapolation_recovery'
    recorded_params = planned['current']['action_records'][0]['final_action']['parameters']
    recorded_params['direct_target'] = 100
    with pytest.raises(ValueError, match='direct_insufficient_verified_candidates_for_approved_target'):
        prepare_direct_sampling(action=direct_action,
            context={'event_state':planned['current'],'effective_config':config,'manager':manager})
    recorded_params.pop('direct_target')
    sampled = prepare_direct_sampling(action=direct_action,
        context={'event_state':planned['current'],'effective_config':config,'manager':manager})
    assert sampled['direct_sampling']['reference_count'] == 1
    fixed, _ = quarantine_refresh(sampled['state'], 'epoch_refresh')
    assert fixed['phase_diagrams']['mlip'] == old
    selected = sampled['direct_sampling']['selected'][0]
    from phase_agent.tools.workflows.perturbation_direct import previously_approved_recovery_action
    dft_action = {'tool':'select_dft_candidates', 'parameters':{'decisions':[{'candidate_id':selected['candidate_id'], 'action':'DFT_SINGLE_POINT'}]}}
    assert previously_approved_recovery_action(fixed, dft_action)
    dft_action['parameters']['decisions'][0]['candidate_id'] = 'unapproved'
    assert not previously_approved_recovery_action(fixed, dft_action)
    from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
    from phase_agent.tools.workflows.create_dft_selection_handler import create_dft_selection_handler
    config.update(qbc=default_dft_decision_config(), mlip={'version':'mace-mh-1'}, dft={'parameters':{}})
    dft_action['parameters']['decisions'][0]['candidate_id'] = selected['candidate_id']
    dft_action['parameters']['decisions'][0]['reason'] = 'selected DIRECT representative'
    selected_result = create_dft_selection_handler()(action=dft_action, context={'event_state':fixed,
        'manager':manager,'effective_config':config,'config_version':'cfg','execution_mode':'autonomous'})
    assert selected_result['status'] == 'completed', selected_result
    fixed = selected_result['state']
    dft_task = selected_result['tasks'][0]
    assert dft_task['model_version'] == 'm'
    dft_id = dft_task['task_id']
    for task in fixed['tasks']:
        if task['task_id'] == dft_id:
            task['status'] = 'completed'
    monkeypatch.setattr(spin_acceptance, 'spin_standard_passed', lambda record: True)
    fixed['dft_dataset_records'].append({'task_id':dft_id,'status':'completed','converged':True,'checks_passed':True,'training_ready':True,'final_frame_valid':True})
    assert recovery_label_facts(fixed)['status'] == 'complete'
    training_action = {'tool':'update_mlip','operation':'prepare_mlip_training',
                       'parameters':{'action':'RETRAIN_MLIP','prepare_inputs_only':True}}
    assert previously_approved_recovery_action(fixed, training_action)
    trained = _prepare_synthetic_recovery_training(fixed, dft_id, selected['structure_path'], tmp_path)
    job = next(iter(trained['remote_finetune_jobs'].values()))
    assert dft_id + ':frame:0' in job['report']['dataset_scope']['data_ids']
    job.update(status='results_received', training_handoff={'candidate_model_version':'m-next'})
    assert recovery_model_matches_labels(trained, 'm-next')


def test_synthetic_extrapolation_route_reuses_approval_for_dft_and_training(tmp_path, monkeypatch):
    state, context, action, frame = make_case(tmp_path, monkeypatch)
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh, recovery_label_facts, recovery_model_matches_labels
    from phase_agent.tools.workflows.perturbation_direct import previously_approved_recovery_action
    from phase_agent.science.dft import spin_acceptance
    old = {'status':'completed','version':'old-hull','model_version':'old','entries':[]}
    state['phase_diagrams'] = {'mlip':{'status':'completed','version':'bad-hull','model_version':'m','entries':[]}}
    state['model_refresh'].update(old_diagram=old, old_model_version='old')
    digest = assess_current_rounds(state)['rounds'][0]['evidence_digest']
    action['parameters']['round_evidence_digest'] = digest
    action['operation'] = 'sample_direct_extrapolation'
    action['tool'] = 'prepare_local_batch_files'
    action['parameters']['mode'] = 'direct_extrapolation'
    state['action_records'] = [{'record_id':'route-approval','status':'completed','human_feedback':{'decision':'approve'},'final_action':action}]
    context['approval_record_id'] = 'route-approval'
    sampled = prepare_direct_sampling(action=action, context=context)
    assert sampled['direct_sampling']['reference_count'] == 1
    fixed, _ = quarantine_refresh(sampled['state'], 'epoch_refresh')
    selected = sampled['direct_sampling']['selected'][0]
    assert fixed['phase_diagrams']['mlip'] == old
    dft = {'tool':'select_dft_candidates','parameters':{'decisions':[{'candidate_id':selected['candidate_id'],'action':'DFT_SINGLE_POINT'}]}}
    assert previously_approved_recovery_action(fixed, dft)
    monkeypatch.setattr(spin_acceptance, 'spin_standard_passed', lambda record:True)
    fixed['tasks'].append({'task_id':'new-dft','stage':'dft_single_point','status':'completed','structure_path':selected['structure_path']})
    fixed['dft_dataset_records'].append({'task_id':'new-dft','status':'completed','converged':True,'checks_passed':True,'training_ready':True,'final_frame_valid':True})
    assert recovery_label_facts(fixed)['status'] == 'complete'
    train = {'tool':'update_mlip','operation':'prepare_mlip_training','parameters':{'action':'RETRAIN_MLIP','prepare_inputs_only':True}}
    assert previously_approved_recovery_action(fixed, train)
    trained = _prepare_synthetic_recovery_training(fixed, 'new-dft', selected['structure_path'], tmp_path)
    job = next(iter(trained['remote_finetune_jobs'].values()))
    assert 'new-dft:frame:0' in job['report']['dataset_scope']['data_ids']
    job.update(status='results_received', training_handoff={'candidate_model_version':'m-next'})
    assert recovery_model_matches_labels(trained, 'm-next')


def _prepare_synthetic_recovery_training(state, task_id, structure_path, tmp_path):
    """Build the real training input bundle from synthetic same-frame DFT labels."""
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    from phase_agent.analysis.feedback.dft_result_products import record_dft_products
    from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    atoms = read(structure_path, index=0)
    structure = AseAtomsAdaptor.get_structure(atoms)
    result = {'task_id':task_id, 'structure_id':'recovery-structure', 'branch_id':'recovery-branch',
        'model_version':'m', 'stage':'dft_single_point', 'status':'completed', 'converged':True,
        'checks_passed':True, 'outputs':{'structure':structure.as_dict(), 'composition':structure.composition.as_dict(),
        'energy':-8.0, 'training_energy':-8.0, 'energy_unit':'eV', 'forces':[[0.1,0.0,0.0]]*len(atoms),
        'forces_unit':'eV/angstrom', 'atom_count':len(atoms), 'training_ready':True,
        'training_frame_index':0, 'final_frame_index':0, 'final_frame_valid':True}}
    record_dft_products(state, result)
    for index in range(10):
        historic = deepcopy(result)
        historic['task_id'] = 'historic-' + str(index)
        historic['structure_id'] = 'history-' + str(index)
        historic['branch_id'] = 'history-branch-' + str(index)
        state['tasks'].append({'task_id':historic['task_id'], 'stage':'dft_single_point',
                               'status':'completed', 'model_version':'m'})
        record_dft_products(state, historic)
    state['active_model'] = {'version':'m', 'model_path':'/remote/failed.model'}
    state['confirmed_config'] = {'mlip':{'model_path':'/remote/original-mh1.model'}}
    settings = default_mace_committee_config()
    settings['labels']['include_stress'] = False
    training = prepare_remote_finetune(state, {'mlip_finetune':settings,
        'upload_batches_directory':str(tmp_path/'uploads'), 'mlip':state['active_model'],
        'python_environments':{'remote_mlip':'mace'}})
    assert training['status'] == 'awaiting_remote_training', training
    handoff = training['handoffs'][0]
    assert Path(handoff['upload_archive']).is_file() and not training['submitted']
    return training['state']
