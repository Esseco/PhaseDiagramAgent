from pathlib import Path
import json
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.runtime.diagram_projects import migrate


def test_diagram_migration_moves_real_directories_without_aliases(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir()
    config=default_layered_search_config();config['storage']=default_workspace_storage(root);config['system']['boundary']={'P':['O3'],'H':{'max_det':12},'TM_ratio':{'Fe':1,'Mn':1}}
    state={'confirmed_config':config,'tasks':[],'confirmed_config_version':'same'}
    (root/'workflow_state').mkdir();(root/'parameters').mkdir();(root/'logs').mkdir()
    (root/'workflow_state/state.json').write_text(json.dumps(state))
    (root/'parameters/config_session.json').write_text('{}')
    (root/'agent_runtime.json').write_text(json.dumps({'state_path':str(root/'workflow_state/state.json'),'ledger_path':str(root/'workflow_state/ledgers/phase_data.json')}))
    manager=PhaseDataManager(config['system']['boundary'],system_config=config['system']);manager.save(root/'workflow_state/ledgers/phase_data.json')
    monkeypatch.setattr('phase_agent.runtime.diagram_projects.publish',lambda *a:0)
    migrate(root,backup_directory=tmp_path/'backup')
    assert not (root/'workflow_state').exists()
    assert (root/'diagrams/diagram_default/workflow_state/state.json').is_file()
    assert not any(p.is_junction() or p.is_symlink() for p in root.rglob('*'))
    runtime=json.loads((root/'agent_runtime.json').read_text())
    assert Path(runtime['state_path']).is_file()
    assert Path(runtime['editable_config_draft_path']).parent==root/'diagrams/diagram_default/parameters'
