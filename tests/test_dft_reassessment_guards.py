from copy import deepcopy
import json
from pathlib import Path
import pytest
from tests.test_round_model_recovery import dft_result
from phase_agent.tools.remote.dft_result_reassessment import reassessment_transform
from phase_agent.tools.remote.integrity import file_checksum

@pytest.fixture
def candidate(tmp_path):
    source=tmp_path/'source'
    worker=source/'worker'
    worker.mkdir(parents=True)
    row=dft_result()
    row.update(status='failed',converged=False,error='vasp exited with code 1',task_checksum='b'*64)
    row['outputs'].update(xml_complete=True,electronic_converged=True)
    from pymatgen.core import Structure
    structure=Structure.from_dict(row['outputs']['structure'])
    structure.to(filename=worker/'initial.vasp',fmt='poscar')
    model=tmp_path/'weights.model'
    model.write_text('synthetic test model')
    comparison={'version':'m1','model_path':str(model),'environment':'current'}
    (worker/'comparison_model.json').write_text(json.dumps(comparison))
    (worker/'workflow.json').write_text(json.dumps({'calculation':'static','structure_file':'initial.vasp'}))
    (worker/'workflow_state.json').write_text(json.dumps({'stages':{'static':{'error':'static unexpectedly changed the structure','directory':'runs/static/attempt_001'}}}))
    (worker/'runs/static/attempt_001').mkdir(parents=True)
    dest=tmp_path/'derived'
    dest.mkdir()
    (dest/'result.json').write_text(json.dumps(row))
    expected={key:row[key] for key in ['task_id','task_key','model_version','task_checksum']}
    (dest/'task.finished.json').write_text(json.dumps({**expected,'status':'failed','result_checksum':file_checksum(dest/'result.json')}))
    options={'model':comparison,'eligible_tasks':{row['task_id']:{'original_result_sha256':file_checksum(dest/'result.json'),'initial_structure_sha256':file_checksum(worker/'initial.vasp'),'comparison_model_sha256':file_checksum(worker/'comparison_model.json')}}}
    work=tmp_path/'work'
    work.mkdir()
    return source,worker,dest,expected,row,structure,options,work

@pytest.mark.parametrize('defect',['result','initial','workflow','geometry','xml','final_frame'])
def test_refuses_ineligible_or_changed_data_before_prediction(candidate,defect):
    source,worker,dest,expected,row,structure,options,work=candidate
    final=deepcopy(structure)
    if defect=='result':(dest/'result.json').write_text('{}')
    elif defect=='initial':(worker/'initial.vasp').write_text('changed')
    elif defect=='workflow':(worker/'workflow_state.json').write_text(json.dumps({'stages':{'static':{'error':'UnconvergedErrorHandler','directory':'runs/static/attempt_001'}}}))
    elif defect=='geometry':final.translate_sites([0],[.05,0,0])
    elif defect=='final_frame':row['outputs']['final_frame_valid']=False
    def validator(*args):
        if defect=='xml':raise ValueError('XML incomplete or not electronically converged')
        return final
    def predictor(*args,**kwargs):raise AssertionError('Invalid result must not invoke prediction')
    transform=reassessment_transform(work,options,validator=validator,predictor=predictor)
    with pytest.raises(ValueError):transform(dest,source,{'input_directory':'worker','expected':expected},row)

def test_prediction_failure_preserves_valid_dft_and_original_failure(candidate):
    source,worker,dest,expected,row,structure,options,work=candidate
    original_bytes=(dest/'result.json').read_bytes()
    transform=reassessment_transform(work,options,validator=lambda *args:structure,predictor=lambda *args,**kwargs:{'status':'failed','error':'synthetic GPU failure'})
    report=transform(dest,source,{'input_directory':'worker','expected':expected},row)
    revised=json.loads((dest/'result.json').read_text())
    assert report['status']=='prediction_failed'
    assert revised['status']=='completed' and revised['outputs']['energy']==row['outputs']['energy']
    assert (dest/'original_result.json').read_bytes()==original_bytes
    assert revised['recovery_provenance']['original_status']=='failed'
