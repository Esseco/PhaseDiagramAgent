"""Bind externally submitted DFT retry results to registered failed tasks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def register_diagnostic_retry(state, input_archive, result_archive, evidence_dir):
    """Validate every frame and terminal receipt before registering pending retries."""
    from phase_agent.tools.remote.input_validation import _inventory
    from phase_agent.tools.remote.integrity import payload_checksum, verify_result
    from phase_agent.tools.remote.transfer_bundle import checksum, import_bundle, validate_bundle

    source, returned = Path(input_archive), Path(result_archive)
    source_sha, result_sha = checksum(source), checksum(returned)
    validate_bundle(returned)
    existing_imports = state.get('diagnostic_retry_imports') or {}
    if source_sha in existing_imports:
        if existing_imports[source_sha].get('result_sha256') != result_sha:
            raise ValueError('diagnostic_retry_import_conflict')
        # Rebuild only the two immutable input files needed by the retry handler.
        registered={r.get('task_id'):r for r in state.get('tasks') or []}
        with zipfile.ZipFile(source) as source_zip:
            inventory={row['path']:row for row in _inventory(source_zip)}
            scripts=json.loads(source_zip.read('submission_plan.json')).get('scripts') or []
            if len(scripts)!=len(existing_imports[source_sha]['task_ids']):
                raise ValueError('diagnostic_retry_existing_inventory_invalid')
            for script in scripts:
                folder=PurePosixPath(script).parent
                task=json.loads(source_zip.read(str(folder/'task.json')))
                saved=registered.get(task.get('task_id')) or {}
                if (task.get('task_id') not in existing_imports[source_sha]['task_ids']
                        or task.get('task_checksum')!=saved.get('task_checksum')):
                    raise ValueError('diagnostic_retry_existing_task_invalid')
                dest=Path(saved['input_path']).parent
                for name in ('workflow.json','submit_gpu.sh'):
                    entry=str(folder/name); row=inventory[entry]; data=source_zip.read(entry)
                    if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                        raise ValueError('diagnostic_retry_input_checksum_invalid')
                    target=dest/name
                    if target.exists() and target.read_bytes()!=data:
                        raise ValueError('diagnostic_retry_evidence_conflict')
                    if not target.exists(): target.write_bytes(data)
        return {'status':'already_registered', 'state':deepcopy(state), 'task_count':len(existing_imports[source_sha]['task_ids'])}
    original = {r.get('task_id'):r for r in state.get('tasks') or []}
    tasks=[]
    with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(returned) as result_zip:
        rows=_inventory(source_zip); files={r['path'] for r in rows}
        for row in rows:
            data=source_zip.read(row['path'])
            if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('diagnostic_retry_input_checksum_invalid')
        plan=json.loads(source_zip.read('submission_plan.json')); scripts=plan.get('scripts') or []
        if not scripts or len(scripts)!=len(set(scripts)):
            raise ValueError('diagnostic_retry_plan_invalid')
        transfer=json.loads(result_zip.read('transfer_manifest.json'))
        if (transfer.get('batch_binding') or {}).get('input_sha256') != source_sha:
            raise ValueError('diagnostic_retry_result_binding_invalid')
        allowed={'results/'+PurePosixPath(s).parent.name+'/'+name for s in scripts
                 for name in ('result.json','task.finished.json','training.json')}
        result_names={r['path'] for r in transfer['files']}
        if not result_names.issubset(allowed):
            raise ValueError('diagnostic_retry_unexpected_result')
        with tempfile.TemporaryDirectory(prefix='diagnostic-retry-') as tmp:
            temp=Path(tmp)
            for script in scripts:
                folder=PurePosixPath(script).parent; task_file=str(folder/'task.json')
                if task_file not in files: raise ValueError('diagnostic_retry_task_missing')
                task=json.loads(source_zip.read(task_file)); old_id=task.get('retry_of'); old=original.get(old_id) or {}
                signed=task.pop('task_checksum',None)
                if (signed != payload_checksum(task) or old.get('status') not in {'failed','timeout'}
                    or task.get('task_id') in original or task.get('stage') != 'dft_single_point'
                    or task.get('structure_id') != old.get('structure_id')
                    or task.get('input_structure_sha256') != old.get('input_structure_sha256')
                    or task.get('model_version') != old.get('model_version')
                    or task.get('config_version') != old.get('config_version')
                    or task.get('config_version') != state.get('confirmed_config_version')
                    or task.get('model_version') != state.get('active_model_version')
                    or not task.get('task_key','').startswith(old.get('task_key','')+':')):
                    raise ValueError('diagnostic_retry_task_binding_invalid')
                data=source_zip.read(str(folder/'initial_structure.vasp'))
                if hashlib.sha256(data).hexdigest()!=old.get('input_structure_sha256'):
                    raise ValueError('diagnostic_retry_structure_changed')
                actual=hashlib.sha256(Path(old['structure_path']).read_bytes()).hexdigest()
                if actual!=old.get('input_structure_sha256'):
                    raise ValueError('diagnostic_retry_parent_structure_changed')
                wf=json.loads(source_zip.read(str(folder/'workflow.json')))
                overrides=(task.get('parameters') or {}).get('incar_settings',{}).get('static',{})
                if not overrides or any(wf['incar_settings']['static'].get(k)!=v for k,v in overrides.items()):
                    raise ValueError('diagnostic_retry_settings_mismatch')
                prefix='results/'+folder.name+'/'
                if not {prefix+'result.json',prefix+'task.finished.json'}.issubset(result_names):
                    raise ValueError('diagnostic_retry_terminal_result_missing')
                work=temp/folder.name; work.mkdir()
                for name in ('result.json','task.finished.json','training.json'):
                    if prefix+name in result_names: (work/name).write_bytes(result_zip.read(prefix+name))
                check=verify_result(work/'result.json',work/'task.finished.json',{**task,'task_checksum':signed})
                if not check.get('valid'): raise ValueError('diagnostic_retry_result_invalid:'+str(check.get('reason')))
                record=check['result']
                if record.get('status') not in {'completed','failed'}: raise ValueError('diagnostic_retry_result_nonterminal')
                if record.get('status')=='completed' and not (work/'training.json').is_file():
                    raise ValueError('diagnostic_retry_training_missing')
                task['task_checksum']=signed
                task.update(status='pending',structure_path=old['structure_path'],
                    input_path=str(Path(evidence_dir)/'tasks'/folder.name/'task.json'),
                    result_path=str(Path(evidence_dir)/'results'/folder.name/'result.json'),
                    upload_operation_id='diagnostic-retry-'+source_sha[:16])
                tasks.append(task)
    if len({t['retry_of'] for t in tasks})!=len(tasks): raise ValueError('diagnostic_retry_duplicate_parent')
    evidence=Path(evidence_dir); import_bundle(returned,evidence)
    for task,script in zip(tasks,scripts):
        folder=evidence/'tasks'/PurePosixPath(script).parent.name; folder.mkdir(parents=True,exist_ok=True)
        source_task=dict(task); source_task.pop('input_path',None); source_task.pop('result_path',None); source_task.pop('upload_operation_id',None)
        source_task['structure_path']='initial_structure.vasp'
        data=json.dumps(source_task,ensure_ascii=False,indent=2).encode()
        target=folder/'task.json'
        if target.exists() and target.read_bytes()!=data: raise ValueError('diagnostic_retry_evidence_conflict')
        if not target.exists(): target.write_bytes(data)
        with zipfile.ZipFile(source) as source_zip:
            for name in ('workflow.json','submit_gpu.sh'):
                entry=str(PurePosixPath(script).parent/name); data=source_zip.read(entry); target=folder/name
                if target.exists() and target.read_bytes()!=data: raise ValueError('diagnostic_retry_evidence_conflict')
                if not target.exists(): target.write_bytes(data)
    current=deepcopy(state); current.setdefault('tasks',[]).extend(deepcopy(tasks)); current.setdefault('pending_tasks',[]).extend(deepcopy(tasks))
    current.setdefault('diagnostic_retry_imports',{})[source_sha]={'result_sha256':result_sha,'task_ids':[t['task_id'] for t in tasks]}
    return {'status':'registered','state':current,'task_count':len(tasks),'input_sha256':source_sha,'result_sha256':result_sha}
