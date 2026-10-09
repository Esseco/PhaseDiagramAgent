"""Advance manual HPC handoffs from durable returned artifacts, without submitting."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from execution_layer.remote.write_training_submission import training_environment
from execution_layer.remote.write_unix_shell_script import write_unix_shell_script
from scientific_layer.training.validate_mlip import REQUIRED_LIMITS, validate_mlip


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_job(inputs, filename, command, config):
    import shlex
    template = config.get('remote_training_job_template')
    template = Path(template) if template else Path(__file__).parents[1] / 'remote/mlip_gpu_template.sh'
    script = template.read_text(encoding='utf-8')
    entry = 'python3 run_mlip_task.py'
    if script.count(entry) != 1 or script.count('conda activate mace') != 1:
        raise ValueError('GPU模板需唯一的python3 run_mlip_task.py及conda activate mace入口')
    environment, _ = training_environment(config)
    activation = '' if environment == 'current' else 'conda activate ' + shlex.quote(environment)
    script = script.replace('conda activate mace', activation).replace(entry,
        'set -euo pipefail\ncd "${SLURM_SUBMIT_DIR:-.}"\n' + command)
    script = script.replace('__JOB_NAME__', 'mlip-' + filename.removesuffix('.sh'))
    path = inputs / filename
    # Repeated continue preserves the exact submission artifact, including mtime.
    if not path.exists() or path.read_text(encoding='utf-8') != script:
        write_unix_shell_script(path, script)
    return str(path)


def prepare_manifest_job(job, config, handoff):
    inputs = Path(job["directory"]) / "inputs"
    if not inputs.is_dir() or not (inputs / 'training_plan.json').is_file():
        raise ValueError('本地inputs或training_plan.json缺失；请恢复原训练输入')
    source = Path(__file__).parents[1] / 'remote/collect_training_results.py'
    target = inputs / source.name
    if not target.exists() or target.read_bytes() != source.read_bytes():
        target.write_bytes(source.read_bytes())
    script = write_job(inputs, 'GPU_manifest.sh', 'python collect_training_results.py --manifest-only', config)
    handoff.update(stage='awaiting_manifest', script=script,
        reason=f'上传更新的collect_training_results.py和GPU_manifest.sh，在超算原inputs目录执行 sbatch GPU_manifest.sh；回传同轮results/models.json后说“继续”。所有运算均通过计算节点，未重新训练。')

    return handoff

def validation_request(current, key, job, config, result):
    inputs = Path(job["directory"]) / "inputs"
    settings = config.get('remote_training_validation') or {}
    old = deepcopy(current.get('active_model') or config.get('mlip') or {})
    new = deepcopy(next(row for row in result['models'] if row['is_main_model']))
    new['model_path'] = new.get('remote_model_path') or new['model_path']
    new['version'] = 'remote-' + key + '-' + result['fingerprint'][:12]
    new['dataset_version'] = job['report'].get('training_round', Path(job['directory']).name)
    required = []
    data = Path(settings.get('data_path') or '__missing_validation_data__')
    if not data.is_file():
        required.append('未参与本轮训练的DFT独立验证extxyz文件(data_path)')
    if not settings.get('data_version'):
        required.append('独立验证数据版本(data_version)')
    criteria = settings.get('criteria') or (config.get('mlip_finetune') or {}).get('validation') or {}
    for name in REQUIRED_LIMITS:
        if criteria.get(name) is None:
            required.append('验证标准criteria.' + name)
    pairs = settings.get('ranking_pairs')
    if not isinstance(pairs, list) or not pairs:
        required.append('同组成近凸包结构索引对ranking_pairs')
    if not PurePosixPath(old.get('model_path') or '').is_absolute() or len(old.get('sha256') or '') != 64:
        required.append('旧模型的超算绝对路径及SHA256')
    if required:
        return {"missing": required}
    for value in criteria.values():
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError('验证阈值必须为有限非负数')
    training = inputs / '_shared_data/train.xyz'
    plan = {'training_fingerprint': result['fingerprint'], 'data_sha256': digest_file(data),
        'training_sha256': digest_file(training), 'data_version': settings['data_version'],
        'old_model': old, 'new_model': new, 'criteria': criteria, 'ranking_pairs': pairs,
        'energy_key': settings.get('energy_key', 'REF_energy'),
        'forces_key': settings.get('forces_key', 'REF_forces')}
    request_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:24]
    plan['request_id'] = request_id
    returned = Path(job['directory']) / 'results' / ('validation-' + request_id + '.json')
    return {"plan": plan, "returned": str(returned), "new": new, "old": old, "criteria": criteria, "settings": settings}

def recover_validation(current, request, handoff):
    returned = Path(request["returned"])
    plan, new, old, criteria, settings = (request[k] for k in ("plan", "new", "old", "criteria", "settings"))
    request_id = plan["request_id"]
    report = json.loads(returned.read_text(encoding='utf-8-sig'))
    if (report.get('request_id') != request_id or report.get('data_sha256') != plan['data_sha256']
            or report.get('status') != 'completed' or type(report.get('structures')) is not int or report['structures'] < 1):
        raise ValueError('独立验证回传身份、状态或结构数不匹配')
    for metrics in report['metrics'].values():
        if any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0 for v in metrics.values()):
            raise ValueError('验证回传含非法指标')
    validation = validate_mlip(old, new, [], criteria=criteria,
        validation_data_version=settings['data_version'],
        evaluator=lambda model, data: report['metrics']['new_model' if model is new else 'old_model'])
    stored = current.setdefault('candidate_models', {}).get(new['version'])
    if not stored or stored.get('status') != 'user_rejected':
        current['candidate_models'][new['version']] = {'model': new, 'validation': validation,
            'old_model_version': old.get('version'), 'dft_data_version': new['dataset_version'],
            'validation_data_version': settings['data_version'], 'status': 'validated_candidate'}
    stage = ('candidate_rejected' if stored and stored.get('status') == 'user_rejected' else
             'awaiting_activation_approval' if validation.get('passed') else 'validation_rejected')
    handoff.update(stage=stage, validation=validation, validation_sha256=digest_file(returned),
        reason=f'独立验证已回收，候选{new["version"]}；状态{stage}。验证摘要：{json.dumps(validation, ensure_ascii=False)}。单独审阅后可回复“激活候选 {new["version"]} 原因：你的理由”或“拒绝候选 {new["version"]} 原因：你的理由”；“继续”不会激活。')

    return current, handoff

def prepare_validation_job(job, config, request, handoff):
    inputs = Path(job["directory"]) / "inputs"
    plan = request["plan"]
    settings = request["settings"]
    data = Path(settings["data_path"])
    returned = Path(request["returned"])
    inputs.mkdir(exist_ok=True)
    for filename, content in [('validation.xyz', data.read_bytes()),
            ('validation_request.json', json.dumps(plan, indent=2).encode()),
            ('validate_remote_training.py', (Path(__file__).parents[1] / 'remote/validate_remote_training.py').read_bytes())]:
        target = inputs / filename
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    script = write_job(inputs, 'GPU_validation.sh', 'python validate_remote_training.py', config)
    handoff.update(stage='awaiting_validation', script=script,
        reason=f'独立验证作业已生成：{script}；上传validation.xyz、validation_request.json、validate_remote_training.py和GPU_validation.sh到原inputs，执行 sbatch GPU_validation.sh。回传results/{returned.name}后继续；无需重新训练。')

    return handoff

def record_handoff(current, key, handoff):
    job = current['remote_finetune_jobs'][key]
    prior = job.get('training_handoff') or {}
    job['training_handoff'] = deepcopy(handoff)
    if prior != handoff:
        current.setdefault('training_handoff_history', []).append({'job_key': key, **deepcopy(handoff)})
    return current


def advance_training_handoffs(state, config, *, state_path=None, graph=None):
    from orchestration.training_handoff_graph import run_training_handoffs
    return run_training_handoffs(state, config, state_path=state_path, graph=graph)
