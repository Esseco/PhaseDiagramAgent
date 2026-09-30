"""Refresh unstarted DFT input files without recreating tasks or reservations."""
from copy import deepcopy
from pathlib import Path
import json
import re
import shutil
import tempfile
from datetime import datetime


def regenerate_dft_files(state, message, *, manager, upload_root):
    from execution_layer.local.identify_rerun_plan import identify_rerun_plan
    from execution_layer.remote.batch_runner import _task_directories_by_id, _locate_task_directory
    from scientific_layer.dft.prepare_pycode_relax import prepare_pycode_relax
    plan = identify_rerun_plan(message, state)
    if not plan or plan.get('status') != 'identified' or plan.get('action', {}).get('tool') != 'select_dft_candidates':
        raise ValueError('无法唯一定位批准的 DFT 批次')
    import hashlib
    operation = hashlib.sha256(str(plan['action']['task_key'] or plan['action']['record_id']).encode()).hexdigest()[:12]
    tasks = [row for row in state.get('tasks') or [] if row.get('stage') in {'dft_single_point','dft_relax'}
             and (row.get('upload_operation_id') == operation or row.get('parent_decision_id') == plan['action']['record_id'])]
    if not tasks or len(tasks) != plan['affected_task_count']:
        raise ValueError('任务范围不一致')
    root = Path(upload_root).resolve()
    indexed = _task_directories_by_id(state)
    selected = []
    for task in tasks:
        directory, snapshot, error = _locate_task_directory(task, indexed)
        if error or directory is None:
            raise ValueError('任务目录不明确：' + task['task_id'])
        directory = directory.resolve()
        if root not in directory.parents:
            raise ValueError('任务目录超出上传范围')
        if task.get('status') not in {'pending','prepared'}:
            raise ValueError('任务可能已经执行：' + task['task_id'])
        if any((directory/name).exists() for name in ['runs','workflow_state.json','result.json','task.finished.json','out','err']):
            raise ValueError('存在运行记录或结果，不覆盖：' + str(directory))
        result_path = task.get('result_path')
        if result_path and Path(result_path).is_file():
            raise ValueError('已有回传结果，不覆盖')
        selected.append((task,directory,snapshot))
    backup_root = root.parent/'dft_input_backups'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    # Generate everything before changing any existing input directory.
    with tempfile.TemporaryDirectory(prefix='dft-rebuild-') as temporary:
        staged = []
        for task,directory,snapshot in selected:
            target = Path(temporary)/task['task_id']
            request = {**deepcopy(snapshot),'work_directory':str(target)}
            prepare_pycode_relax(request,manager=manager)
            script = task.get('reviewed_submit_script') or snapshot.get('reviewed_submit_script')
            if script:
                script = re.sub(r'(?m)^#SBATCH --job-name=.*$', '#SBATCH --job-name=DFT-'+task['task_id'],script)
                (target/'submit_gpu.sh').write_text(script,encoding='utf-8',newline='\n')
            shutil.copyfile(target/'submit_gpu.sh',target/'GPU.sh')
            staged.append((task,directory,target))
        changed = []
        try:
            for task,directory,target in staged:
                backup = backup_root/task['task_id']
                backup.mkdir(parents=True)
                names = []
                for path in target.iterdir():
                    if path.is_file():
                        old = directory/path.name
                        existed = old.is_file()
                        if existed: shutil.copy2(old,backup/path.name)
                        names.append((path.name,existed))
                changed.append((directory,backup,names))
                for name,_ in names: shutil.copy2(target/name,directory/name)
        except Exception:
            for directory,backup,names in reversed(changed):
                for name,existed in names:
                    if existed: shutil.copy2(backup/name,directory/name)
                    else: (directory/name).unlink(missing_ok=True)
            raise
    return {'task_count':len(selected),'backup_directory':str(backup_root),'submitted':False}
