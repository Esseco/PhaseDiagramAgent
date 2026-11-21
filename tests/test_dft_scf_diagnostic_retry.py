import json
import pytest

from phase_agent.science.dft.incar_policy import layered_oxide_incar
from phase_agent.tools.workflows.dft_failure_retry import scf_diagnostic_retry_settings
from phase_agent.tools.workflows import create_workflow_handlers as handlers


def test_retry_settings_reach_incar_without_overwriting_mixing(tmp_path):
    (tmp_path / 'workflow.json').write_text(json.dumps({'incar_settings': {'static': {'NELM': 150, 'ENCUT': 520}}}), encoding='utf8')
    (tmp_path / 'submit_gpu.sh').write_text('python3 workflow.py\n', encoding='utf8')
    original = {'stage': 'dft_single_point', 'input_path': str(tmp_path / 'task.json')}
    normal, _ = scf_diagnostic_retry_settings(original, 'nelm_200')
    slow, _ = scf_diagnostic_retry_settings(original, 'nelm_200_slow_mixing')
    normal = layered_oxide_incar(normal['incar_settings']['static'], static=True)
    slow = layered_oxide_incar(slow['incar_settings']['static'], static=True)
    assert (normal['NELM'],normal['AMIX'],normal['AMIX_MAG'],normal['BMIX'],normal['BMIX_MAG']) == (200,.2,.8,.0001,.0001)
    assert (slow['NELM'],slow['AMIX'],slow['AMIX_MAG'],slow['BMIX'],slow['BMIX_MAG']) == (200,.1,.4,.0001,.0001)
    assert normal['ENCUT'] == slow['ENCUT'] == 520
    with pytest.raises(ValueError, match='invalid_scf_diagnostic_strategy'):
        scf_diagnostic_retry_settings(original, 'invented')


def test_handler_binds_each_strategy_and_rejects_scope_changes(tmp_path, monkeypatch):
    (tmp_path / 'workflow.json').write_text(json.dumps({'incar_settings': {'static': {'NELM': 150}}}), encoding='utf8')
    (tmp_path / 'submit_gpu.sh').write_text('python3 workflow.py\n', encoding='utf8')
    tasks = [{'task_id': x, 'task_key': x, 'stage':'dft_single_point', 'status':'failed', 'input_path':str(tmp_path/'task.json')} for x in ['a','b']]
    def retry(state, target, **kwargs):
        t={'task_key':target+':retry','task_id':'retry-'+target}
        state['tasks'].append(t)
        return {'status':'prepared','state':state,'task':t}
    monkeypatch.setattr(handlers,'restart_failed_task',retry)
    state={'tasks':tasks}
    params={'repair_profile':'dft_scf_diagnostic','strategies':{'a':'nelm_200','b':'nelm_200_slow_mixing'}}
    result=handlers._restart_failed_task(action={'target_ids':['a','b'],'parameters':params},context={'event_state':state})
    assert len(state['tasks'])==2
    assert [t['parameters']['incar_settings']['static']['AMIX'] for t in result['tasks']]==[.2,.1]
    assert all(t['parameters']['incar_settings']['static']['NELM']==200 for t in result['tasks'])
    with pytest.raises(ValueError,match='invalid_scf_diagnostic_scope'):
        handlers._restart_failed_task(action={'target_ids':['a'],'parameters':params},context={'event_state':state})


def test_generated_workflow_keeps_reviewed_slow_mixing(tmp_path):
    from types import SimpleNamespace
    from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax
    structure=tmp_path/'source.vasp'; structure.write_text('structure',encoding='utf8')
    work=tmp_path/'work'; work.mkdir()
    def generator(directory, source, **kwargs):
        (directory/'workflow.json').write_text(json.dumps({'incar_settings':kwargs['incar_settings']}),encoding='utf8')
        (directory/'workflow.py').write_text('pass\n',encoding='utf8')
    task={'stage':'dft_single_point','structure_id':'s','structure_path':str(structure),'work_directory':str(work),
          'parameters':{'incar_settings':{'static':{'NELM':200,'AMIX':.1,'AMIX_MAG':.4,'BMIX':.0001,'BMIX_MAG':.0001}}}}
    prepare_pycode_relax(task,manager=SimpleNamespace(data={'structures':{}}),generator=generator)
    actual=json.loads((work/'workflow.json').read_text(encoding='utf8'))['incar_settings']['static']
    assert (actual['NELM'],actual['AMIX'],actual['AMIX_MAG'],actual['BMIX'],actual['BMIX_MAG'])==(200,.1,.4,.0001,.0001)


def test_scf_retry_accepts_explicit_seventeen_targets(tmp_path, monkeypatch):
    (tmp_path / "workflow.json").write_text(json.dumps({"incar_settings": {"static": {}}}), encoding="utf8")
    (tmp_path / "submit_gpu.sh").write_text("python3 workflow.py\n", encoding="utf8")
    ids = [f"task-{i}" for i in range(17)]
    tasks = [{"task_id": x, "task_key": x, "stage": "dft_single_point", "status": "failed", "input_path": str(tmp_path / "task.json")} for x in ids]
    def retry(state, target, **kwargs):
        task = {"task_key": target + ":retry", "task_id": "retry-" + target}
        state["tasks"].append(task)
        return {"status": "prepared", "state": state, "task": task}
    monkeypatch.setattr(handlers, "restart_failed_task", retry)
    params = {"repair_profile": "dft_scf_diagnostic", "strategies": {x: "nelm_200_slow_mixing" for x in ids}}
    result = handlers._restart_failed_task(action={"target_ids": ids, "parameters": params}, context={"event_state": {"tasks": tasks}})
    assert len(result["tasks"]) == 17
    assert all(t["parameters"]["incar_settings"]["static"]["AMIX"] == .1 for t in result["tasks"])


def test_policy_accepts_unique_seventeen_scf_targets():
    from phase_agent.policies.validate_tool_action import validate_tool_action
    ids = [f"D{i}" for i in range(17)]
    session = {"status": "confirmed", "confirmed_snapshot": {"config_version": "config-1", "config": {"agent": {"allowed_tools": ["restart_failed_task"]}, "budgets": {"total_relative_cost": 100}, "frozen_parameters": []}}}
    state = {"tasks": [{"task_id": x, "stage": "dft_single_point", "status": "failed"} for x in ids]}
    action = {"tool": "restart_failed_task", "task_key": "retry", "target_ids": ids, "parameters": {"repair_profile": "dft_scf_diagnostic", "strategies": {x: "nelm_200_slow_mixing" for x in ids}}, "budget": 0}
    registry = {"restart_failed_task": {"handler": lambda **_: None}}
    assert "retry_target_not_failed_task" not in validate_tool_action(action, state, session, registry)["errors"]
    action["target_ids"] = ids + [ids[0]]
    assert "retry_target_not_failed_task" in validate_tool_action(action, state, session, registry)["errors"]
