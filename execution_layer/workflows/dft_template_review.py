"""One representative input preview, bound to a reviewed batch by content hash."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(review):
    folder = Path(review['directory'])
    files = review.get('review_files') or ['settings.json', 'INCAR', 'KPOINTS', 'GPU.sh']
    return hashlib.sha256(b''.join(name.encode() + (folder / name).read_bytes() for name in files)).hexdigest()


def validate_review(review, action, config_version):
    if not review or review.get('status') != 'approved':
        raise ValueError('DFT 公共模板尚未确认')
    if review.get('config_version') != config_version or review.get('selection_hash') != selection_hash(action):
        raise ValueError('候选或配置版本变化，请重新确认模板')
    if digest(review) != review.get('digest'):
        raise ValueError('模板文件已修改，请重新预览并确认')
    return json.loads((Path(review['directory']) / 'settings.json').read_text(encoding='utf-8'))


def selection_hash(action):
    return hashlib.sha256(json.dumps((action.get('parameters') or {}).get('decisions'), sort_keys=True).encode()).hexdigest()


def create_review(action, state, config, manager, *, revision=0, settings=None):
    if manager is None:
        raise ValueError('DFT 模板审核缺少结构台账，未生成模板或计算任务')
    from scientific_layer.dft.incar_policy import layered_oxide_incar
    from Process_Vasp.generation import GPU_SCRIPT
    selected = next(row for row in action['parameters']['decisions'] if row['action'] in {'DFT_SINGLE_POINT', 'DFT_RELAX'})
    stage = 'static' if selected['action'] == 'DFT_SINGLE_POINT' else 'relax'
    candidate = next((row for row in state.get('qbc_candidates') or [] if row.get('candidate_id') == selected['candidate_id']), {})
    record = manager.data['structures'][selected['candidate_id']]
    source = Path(candidate.get('structure_path') or record.get('source_path') or '')
    if not source.is_file():
        raise ValueError('示例结构文件不存在')
    root = Path(config.get('state_path') or 'outputs/state.json').resolve().parent
    identity = hashlib.sha256(str(action.get('task_key')).encode()).hexdigest()[:12]
    folder = root / 'dft_template_review' / identity / f'r{revision:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    if settings is None:
        parameters = deepcopy((config.get('dft') or {}).get('parameters') or {})
        settings = {'parameters': parameters, 'submit_script': GPU_SCRIPT.read_text(encoding='utf-8-sig').replace('\r\n','\n')}
        settings['submit_script'] = re.sub(r'(?m)^python3 1\.py\s*$', 'python3 workflow.py', settings['submit_script'])
    settings = deepcopy(settings)
    if not isinstance(settings.get('parameters'), dict) or set(settings['parameters']) - {'incar_settings', 'kpoints_settings'}:
        raise ValueError('仅支持 INCAR/KPOINTS 公共设置')
    for name in ('incar_settings', 'kpoints_settings'):
        if not isinstance(settings['parameters'].get(name, {}), dict):
            raise ValueError(name + ' 必须是按 static/relax 分组的字典')
        if any(key not in {'static', 'relax'} or not isinstance(value, dict)
               for key, value in settings['parameters'].get(name, {}).items()):
            raise ValueError(name + ' 仅支持 static/relax 参数字典')
    script = settings['submit_script']
    if 'python3 workflow.py' not in script or len(re.findall(r'(?m)^#SBATCH --job-name=', script)) != 1:
        raise ValueError('提交模板需包含唯一 job-name 和 python3 workflow.py')
    (folder / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    incar = layered_oxide_incar(settings['parameters'].get('incar_settings', {}).get(stage))
    incar['GGA'] = None
    if stage == 'static':
        incar.update(NSW=0, IBRION=-1)
    request = {'source': str(source), 'stage': stage, 'incar': incar,
               'kpoints': settings['parameters'].get('kpoints_settings', {}).get(stage, {})}
    (folder / 'render_request.json').write_text(json.dumps(request), encoding='utf-8')
    try:
        from atomate2.vasp.sets.core import StaticSetGenerator
        interpreter = sys.executable
    except ImportError:
        interpreter = str(Path(sys.executable).parent.parent / 'atomate2' / 'python.exe')
    subprocess.run([interpreter, str(Path(__file__).resolve()), str(folder)], check=True, capture_output=True, text=True)
    from scientific_layer.dft.prepare_pycode_relax import prepare_pycode_relax
    from execution_layer.remote.integrity import payload_checksum
    task = {'task_id': 'DFT-example-' + identity, 'task_key': 'example:' + identity,
            'structure_id': selected['candidate_id'], 'structure_path': str(source),
            'stage': 'dft_single_point' if stage == 'static' else 'dft_relax',
            'parameters': deepcopy(settings['parameters']), 'work_directory': str(folder),
            'config_version': state.get('confirmed_config_version'),
            'model_version': state.get('active_model_version'), 'protocol_version': 1,
            'input_file_version': 1, 'preview_only': True}
    receipt = prepare_pycode_relax(task, manager=manager)
    task['atomate'] = receipt
    task['task_checksum'] = payload_checksum(task)
    (folder / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'submit_gpu.sh').write_text(script, encoding='utf-8', newline='\n')
    (folder / 'GPU.sh').write_text(script, encoding='utf-8', newline='\n')
    (folder / 'README.md').write_text(
        f'代表结构：{selected["candidate_id"]}；类型：{stage}。\n'
        '仅生成一组完整任务示例，不提交或登记正式计算任务。\n'
        '包括 POSCAR、initial_structure、INCAR、KPOINTS、GPU.sh、submit_gpu.sh、workflow.json、workflow.py、atomate_relax.py、atomate_runner.py、task.json。\n'
        'POTCAR 若本地赝势可用则输出；否则提供 POTCAR.spec 和 POTCAR_STATUS.json，需在超算配置同版本赝势后由 atomate2 生成。\n'
        'INCAR/KPOINTS 来自实际 atomate2 生成器。MAGMOM、元素顺序、U数组及网格随各结构生成。\n'
        '公共参数编辑 settings.json（parameters.incar_settings / kpoints_settings 按 static/relax 分组）；'
        '脚本编辑 submit_script。保存后回复“刷新模板”；也可在聊天中提出修改。\n'
        '直接改 INCAR/KPOINTS/GPU.sh 不会成为批量设置，需回写 settings.json 再刷新。\n'
        '确认前不会生成整批输入；模板修改后需重新确认。\n', encoding='utf-8')
    review = {'directory': str(folder), 'revision': revision, 'status': 'pending',
              'config_version': state.get('confirmed_config_version'), 'selection_hash': selection_hash(action),
              'candidate_id': selected['candidate_id'], 'stage': stage,
              'review_files': sorted(path.name for path in folder.iterdir() if path.is_file() and path.name != 'README.md')}
    review['digest'] = digest(review)
    return review


def gate_template(proposal, state, stored, feedback, config, manager, agent_client):
    action = proposal['raw_action']
    if action.get('tool') != 'select_dft_candidates' or (feedback or {}).get('decision') == 'reject':
        return proposal, False
    old = action.get('dft_template_review')
    decision = (feedback or {}).get('decision')
    if old and feedback is None and digest(old) == old.get('digest'):
        proposal['dft_template_review'] = old
        return proposal, True
    if old and decision == 'approve':
        try:
            review = {**old, 'status': 'approved'}
            validate_review(review, action, state.get('confirmed_config_version'))
            action['dft_template_review'] = review
            proposal['dft_template_review'] = review
            return proposal, False
        except (ValueError, OSError):
            decision = 'refresh'
    settings = None
    if old:
        settings = json.loads((Path(old['directory']) / 'settings.json').read_text(encoding='utf-8'))
        comment = (feedback or {}).get('comment') or ''
        if decision == 'comment' and comment.strip() not in {'继续', '刷新模板', '刷新', '查看模板'}:
            if not callable(agent_client):
                raise ValueError('模型未配置；请编辑 settings.json 后回复刷新模板')
            response = agent_client({'mode': 'revise_dft_template', 'settings': settings, 'user_feedback': comment,
                'instruction': 'Return JSON {settings:{parameters:{incar_settings:{static:{},relax:{}},kpoints_settings:{static:{},relax:{}}},submit_script:string}}. Revise only requested public settings. Preserve all other settings and script commands. Never change candidate selection or run calculations.'})
            settings = response['settings']
    review = create_review(action, state, config, manager,
        revision=int(old['revision']) + 1 if old else 0, settings=settings)
    action['dft_template_review'] = review
    proposal['dft_template_review'] = review
    return proposal, True


if __name__ == '__main__':
    from pymatgen.core import Structure
    from atomate2.vasp.sets.core import StaticSetGenerator, RelaxSetGenerator
    folder = Path(sys.argv[1])
    request = json.loads((folder / 'render_request.json').read_text(encoding='utf-8'))
    generator = StaticSetGenerator if request['stage'] == 'static' else RelaxSetGenerator
    inputs = generator(user_incar_settings=request['incar'], user_kpoints_settings=request['kpoints']).get_input_set(
        Structure.from_file(request['source']), potcar_spec=True)
    inputs['INCAR'].write_file(folder / 'INCAR')
    from pymatgen.io.vasp.inputs import Poscar
    Poscar(Structure.from_file(request['source'])).write_file(folder / 'POSCAR')
    spec = inputs.get('POTCAR.spec') or ''
    (folder / 'POTCAR.spec').write_text(str(spec), encoding='utf-8')
    try:
        actual = generator(user_incar_settings=request['incar'], user_kpoints_settings=request['kpoints']).get_input_set(
            Structure.from_file(request['source']), potcar_spec=False)
        actual['POTCAR'].write_file(folder / 'POTCAR')
        status = {'available': True}
    except Exception as error:
        status = {'available': False, 'reason': f'{type(error).__name__}: {error}',
                  'required': 'Configure matching licensed VASP pseudopotentials on execution host; never submit a placeholder POTCAR.'}
    (folder / 'POTCAR_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    if inputs.get('KPOINTS') is None:
        (folder / 'KPOINTS').write_text('KSPACING mode: see INCAR KSPACING/KGAMMA; no explicit KPOINTS file.\n', encoding='utf-8')
    else:
        inputs['KPOINTS'].write_file(folder / 'KPOINTS')
