"""Bind transported same-frame predictions to existing DFT records."""
from copy import deepcopy
import hashlib
import json
import zipfile


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def import_comparisons(state, input_zip, output_zip):
    from phase_agent.tools.remote.dft_mlip_pair import validate_prediction
    from phase_agent.analysis.feedback.dft_result_products import record_dft_products
    from phase_agent.science.structures.load_result_structure import load_result_structure
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    current=deepcopy(state)
    tasks={t.get('task_id'):t for t in current.get('tasks',[])}
    records={r.get('task_id'):r for r in current.get('dft_dataset_records',[])}
    with zipfile.ZipFile(input_zip) as source, zipfile.ZipFile(output_zip) as returned:
        if len(source.namelist())!=len(set(source.namelist())) or len(returned.namelist())!=len(set(returned.namelist())):
            raise ValueError('comparison_duplicate_archive_member')
        binding=json.loads(source.read('binding.json'))
        if json.loads(returned.read('binding.json'))!=binding:
            raise ValueError('comparison_binding_mismatch')
        version=binding.get('model_version'); model_hash=binding.get('model_sha256')
        registered=((current.get('model_registry') or {}).get(version) or {}).get('comparison_model_sha256')
        active=current.get('active_model') or {}
        actual=registered or (active.get('sha256') if current.get('active_model_version')==version else None)
        if not actual or model_hash!=actual: raise ValueError('comparison_model_mismatch')
        items=binding.get('tasks') or []
        ids=[x.get('task_id') for x in items]
        if not ids or len(ids)!=len(set(ids)) or any(not isinstance(x,str) or not x or '/' in x or '\\' in x or x in ('.','..') for x in ids):
            raise ValueError('comparison_task_identity_invalid')
        expected={'binding.json'}
        for tid in ids: expected.update('results/'+tid+'/'+n for n in ('result.json','task.finished.json','mlip_result.json'))
        if {name for name in returned.namelist() if not name.endswith('/')}!=expected:
            raise ValueError('comparison_output_inventory_invalid')
        predictions={}
        for item in items:
            tid=item['task_id']; prefix='results/'+tid+'/'
            original=source.read(prefix+'result.json'); marker=json.loads(source.read(prefix+'task.finished.json'))
            if _sha(original)!=item['result_sha256'] or marker.get('result_checksum')!=item['result_sha256']:
                raise ValueError('comparison_input_corrupt')
            original=json.loads(original)
            revised_bytes=returned.read(prefix+'result.json'); revised=json.loads(revised_bytes)
            done=json.loads(returned.read(prefix+'task.finished.json'))
            prediction_bytes=returned.read(prefix+'mlip_result.json'); prediction=json.loads(prediction_bytes)
            if done.get('result_checksum')!=_sha(revised_bytes) or revised.get('outputs',{}).get('mlip_result_checksum')!=_sha(prediction_bytes):
                raise ValueError('comparison_return_checksum_invalid')
            restored=deepcopy(revised)
            restored.get('outputs',{}).pop('mlip_result_file',None)
            restored.get('outputs',{}).pop('mlip_result_checksum',None)
            if restored!=original: raise ValueError('comparison_dft_result_changed')
            task=tasks.get(tid) or {}; rec=records.get(tid) or {}
            if (task.get('status')!='completed' or rec.get('status')!='completed'
                or rec.get('training_ready') is not True or rec.get('checks_passed') is not True
                or not spin_standard_passed(rec) or task.get('model_version')!=version
                or original.get('task_id')!=tid or original.get('task_key')!=task.get('task_key')
                or original.get('model_version')!=version):
                raise ValueError('comparison_task_not_eligible')
            structure,digest=load_result_structure(original.get('outputs') or {})
            _,registered_digest=load_result_structure(rec)
            if digest!=registered_digest or original['outputs'].get('energy')!=rec.get('energy'):
                raise ValueError('comparison_dft_frame_mismatch')
            validate_prediction(original,prediction,model_hash)
            predictions[tid]=prediction
        for tid,pred in predictions.items():
            task=deepcopy(tasks[tid]); record=records[tid]
            task.update(status='completed',converged=True,checks_passed=True)
            task['outputs']=deepcopy(record)
            task['outputs']['remote_mlip_prediction']=pred
            record_dft_products(current,task,evaluator=lambda **kwargs: pred,refresh=True)
            comparison=next((r for r in current.get('dft_mlip_comparisons',[]) if r.get('task_id')==tid),{})
            if comparison.get('status')!='completed': raise ValueError('comparison_record_rejected:'+tid+':'+str(comparison.get('reason')))
    return current,{'status':'registered','count':len(predictions),'model_version':version}
