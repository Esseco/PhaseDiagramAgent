import json,hashlib,zipfile,copy,sys
sys.path.insert(0,r'E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram')
import pytest
from phase_agent.tools.workflows.same_frame_comparison_import import import_comparisons
from tests.test_round_model_recovery import dft_result
from phase_agent.science.structures.load_result_structure import load_result_structure

def test_binding_and_tamper(tmp_path,monkeypatch):
 from phase_agent.science.dft import spin_acceptance
 from phase_agent.analysis.feedback import dft_result_products
 monkeypatch.setattr(spin_acceptance,'spin_standard_passed',lambda r:True)
 def recorder(state,task,evaluator=None,**kw):
  pred=evaluator()
  state.setdefault('dft_mlip_comparisons',[]).append({'task_id':task['task_id'],'status':'completed','model_sha256':pred['model_sha256']})
 monkeypatch.setattr(dft_result_products,'record_dft_products',recorder)
 result=dft_result(); tid=result['task_id']; version=result['model_version']; taskkey=result['task_key']; digest='a'*64
 original=json.dumps(result).encode(); original_hash=hashlib.sha256(original).hexdigest()
 marker={'result_checksum':original_hash}
 binding={'model_version':version,'model_sha256':digest,'tasks':[{'task_id':tid,'result_sha256':original_hash}]}
 src=tmp_path/'input.zip'; ret=tmp_path/'output.zip'
 with zipfile.ZipFile(src,'w') as z:
  z.writestr('binding.json',json.dumps(binding)); z.writestr('results/'+tid+'/result.json',original); z.writestr('results/'+tid+'/task.finished.json',json.dumps(marker))
 structure,struct_hash=load_result_structure(result['outputs'])
 pred={'task_id':tid,'task_key':taskkey,'model_version':version,'status':'completed','model_sha256':digest,'structure_sha256':struct_hash,'final_frame_index':result['outputs']['final_frame_index'],'energy':-7.,'energy_unit':'eV','forces':[[0.,0.,0.]]*len(structure),'forces_unit':'eV/angstrom'}
 pb=json.dumps(pred).encode(); revised=copy.deepcopy(result); revised['outputs']['mlip_result_file']='mlip_result.json'; revised['outputs']['mlip_result_checksum']=hashlib.sha256(pb).hexdigest(); rb=json.dumps(revised).encode()
 with zipfile.ZipFile(ret,'w') as z:
  z.writestr('binding.json',json.dumps(binding)); z.writestr('results/'+tid+'/result.json',rb); z.writestr('results/'+tid+'/task.finished.json',json.dumps({'result_checksum':hashlib.sha256(rb).hexdigest()})); z.writestr('results/'+tid+'/mlip_result.json',pb)
 rec={**result['outputs'],'task_id':tid,'status':'completed','checks_passed':True,'training_ready':True}
 state={'active_model_version':version,'active_model':{'sha256':digest},'tasks':[result],'dft_dataset_records':[rec]}
 updated,receipt=import_comparisons(state,src,ret)
 assert receipt['count']==1 and updated['dft_mlip_comparisons'][0]['status']=='completed'
 assert 'dft_mlip_comparisons' not in state
 state['active_model']['sha256']='b'*64
 with pytest.raises(ValueError,match='model_mismatch'): import_comparisons(state,src,ret)
