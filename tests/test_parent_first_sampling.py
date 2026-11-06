import hashlib
import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from phase_agent.science.mlip.parent_first_sampling import run_parent_first_sampling


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_parent_relaxation_precedes_perturbation_and_only_selected_are_exported(tmp_path):
    atoms = Atoms('NaCoNiO2', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [0, 0, 2]],
                  cell=[6, 6, 6], pbc=True)
    parent = tmp_path / 'parent.vasp'
    reference = tmp_path / 'reference.vasp'
    model = tmp_path / 'model'
    write(parent, atoms, format='vasp')
    write(reference, atoms, format='vasp')
    model.write_bytes(b'model')
    events = []
    paths = []

    def relax(job):
        events.append('relax')
        final = Path(job['output_directory']) / 'initial_relaxed.vasp'
        final.parent.mkdir()
        structure = read(job['structure_path'])
        structure.positions[0, 2] += 0.25
        write(final, structure, format='vasp')
        return {'status': 'completed', 'optimizer_converged': True,
                'structure_path': str(final), 'structure_checksum': sha(final),
                'extrapolation_monitor': {'extrapolation_failure': False, 'failures': [],
                                          'final_feature_difference': 0.2}}

    def extract(inputs, encoder, cache):
        events.append('extract')
        paths.extend(inputs)
        cache.write_text('fake', encoding='utf8')

    def sample(cache, output, target, reference_count):
        events.append('direct')
        assert reference_count == 1 and target == 2
        assert len(read(paths[1], index=':')) == 3
        assert read(paths[1], index=0).positions[0, 2] == pytest.approx(0.25)
        output.mkdir()
        (output / 'selected_indices.json').write_text(json.dumps([
            {'source': paths[1], 'frame': 0}, {'source': paths[1], 'frame': 1}]), encoding='utf8')
        return {'n_samples': 2}

    out = tmp_path / 'result'
    result = run_parent_first_sampling([{'record_id': 'r1', 'path': str(parent), 'sha256': sha(parent)}],
        [{'path': str(reference), 'sha256': sha(reference)}], out, model_path=model,
        model_sha256=sha(model), feature_model_path='unused', target=2, frames_per_parent=3,
        relax_fn=relax, encoder_factory=lambda _: None, extract_fn=extract, sample_fn=sample)
    assert events == ['relax', 'extract', 'direct']
    assert result['selected_count'] == 2 and result['candidate_count'] == 3
    assert sorted(p.name for p in out.iterdir()) == ['result.json', 'selected_0000.vasp', 'selected_0001.vasp']
    manifest = json.loads((out / 'result.json').read_text(encoding='utf8'))
    assert all(item['parent_record_id'] == 'r1' and item['sha256'] == sha(out / item['path'])
               for item in manifest['selected'])
    assert not list(tmp_path.glob('.parent-first-*'))


def test_parent_first_rejects_changed_inputs_before_relaxation(tmp_path):
    p = tmp_path / 'input'; p.write_bytes(b'x')
    model = tmp_path / 'model'; model.write_bytes(b'model')
    with pytest.raises(ValueError, match='input_sha256_mismatch'):
        run_parent_first_sampling([{'record_id':'r','path':str(p),'sha256':'bad'}], [], tmp_path/'out',
            model_path=model, model_sha256=sha(model), feature_model_path='unused',
            relax_fn=lambda job: pytest.fail('relax must not run'), encoder_factory=lambda _: None,
            extract_fn=lambda *args: None, sample_fn=lambda *args, **kwargs: None)


def test_feature_checkpoint_is_checked_before_any_parent_relaxation(tmp_path):
    parent = tmp_path / 'parent'; parent.write_bytes(b'parent')
    model = tmp_path / 'model'; model.write_bytes(b'model')
    checkpoint = tmp_path / 'feature'; checkpoint.mkdir()
    (checkpoint / 'weights').write_bytes(b'weights')
    args = ([{'record_id': 'r', 'path': str(parent), 'sha256': sha(parent)}], [], tmp_path / 'out')
    kwargs = dict(model_path=model, model_sha256=sha(model), feature_model_path=checkpoint,
                  feature_model_sha256='0' * 64, target=1,
                  relax_fn=lambda job: pytest.fail('relax must not run'))
    with pytest.raises(ValueError, match='feature_model_sha256_mismatch'):
        run_parent_first_sampling(*args, **kwargs)
    assert not (tmp_path / 'out').exists()
    assert not (tmp_path / 'out.parents').exists()
    (checkpoint / 'weights').unlink()
    with pytest.raises(ValueError, match='feature_model_missing'):
        run_parent_first_sampling(*args, **kwargs)


def test_parent_first_rejects_unmonitored_relaxation(tmp_path):
    atoms = Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]],
                  cell=[6,6,6], pbc=True)
    parent = tmp_path / 'parent.vasp'; write(parent, atoms, format='vasp')
    model = tmp_path / 'model'; model.write_bytes(b'model')
    def relax(job):
        final = Path(job['output_directory']) / 'initial_relaxed.vasp'
        final.parent.mkdir()
        write(final, read(parent), format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(final),
                'structure_checksum':sha(final),'extrapolation_monitor':{'extrapolation_failure':False}}
    out = tmp_path / 'result'
    with pytest.raises(ValueError, match='parent_first_relaxation_unverified'):
        run_parent_first_sampling([{'record_id':'r','path':str(parent),'sha256':sha(parent)}], [], out,
            model_path=model, model_sha256=sha(model), feature_model_path='unused', target=1,
            relax_fn=relax, encoder_factory=lambda _: None, extract_fn=lambda *args: None,
            sample_fn=lambda *args, **kwargs: None)
    assert not out.exists()
    assert not list(tmp_path.glob('.parent-first-*'))


def test_resume_after_direct_failure_does_not_repeat_completed_relaxation(tmp_path):
    atoms = Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]],
                  cell=[6,6,6], pbc=True)
    parent = tmp_path/'parent.vasp'; write(parent, atoms, format='vasp')
    model = tmp_path/'model'; model.write_bytes(b'model')
    calls = []
    def relax(job):
        calls.append('relax')
        final = Path(job['output_directory'])/'initial_relaxed.vasp'; final.parent.mkdir()
        write(final, read(job['structure_path']), format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(final),
                'structure_checksum':sha(final), 'extrapolation_monitor':{'extrapolation_failure':False,
                'failures':[], 'final_feature_difference':0.1}}
    sources = []
    def extract(inputs, encoder, cache):
        sources[:] = inputs; cache.write_text('fake', encoding='utf8')
    def failed(*args, **kwargs):
        raise RuntimeError('direct failed')
    def success(cache, output, target, reference_count):
        output.mkdir(); (output/'selected_indices.json').write_text(
            json.dumps([{'source':sources[0], 'frame':0}]), encoding='utf8')
        return {'n_samples':1}
    args = ([{'record_id':'r','path':str(parent),'sha256':sha(parent)}], [], tmp_path/'out')
    kwargs = dict(model_path=model, model_sha256=sha(model), feature_model_path='unused', target=1,
                  relax_fn=relax, encoder_factory=lambda _:None, extract_fn=extract)
    with pytest.raises(RuntimeError, match='direct failed'):
        run_parent_first_sampling(*args, **kwargs, sample_fn=failed)
    assert not (tmp_path/'out').exists()
    result = run_parent_first_sampling(*args, **kwargs, sample_fn=success)
    assert result['selected_count'] == 1 and calls == ['relax']


def test_all_parents_relax_before_any_perturbation(tmp_path, monkeypatch):
    import Process_MLIP.nepkit_sampling as sampler
    atoms = Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]],
                  cell=[6,6,6], pbc=True)
    first = tmp_path/'p0.vasp'; write(first, atoms, format='vasp')
    atoms.positions[0,2] += 0.1
    second = tmp_path/'p1.vasp'; write(second, atoms, format='vasp')
    model = tmp_path/'model'; model.write_bytes(b'model')
    events, sources = [], []
    original = sampler.perturb_layered_oxide_structure
    def perturb(*args, **kwargs):
        events.append('perturb'); return original(*args, **kwargs)
    monkeypatch.setattr(sampler, 'perturb_layered_oxide_structure', perturb)
    def relax(job):
        events.append('relax'); final = Path(job['output_directory'])/'initial_relaxed.vasp'; final.parent.mkdir()
        write(final, read(job['structure_path']), format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(final),
                'structure_checksum':sha(final), 'extrapolation_monitor':{'extrapolation_failure':False,
                'failures':[], 'final_feature_difference':0.1}}
    def extract(inputs, encoder, cache):
        events.append('extract'); sources[:] = inputs; cache.write_text('fake', encoding='utf8')
    def sample(cache, output, target, reference_count):
        events.append('direct'); output.mkdir(); (output/'selected_indices.json').write_text(
            json.dumps([{'source':sources[0], 'frame':0}]), encoding='utf8'); return {'n_samples':1}
    run_parent_first_sampling([{'record_id':'r0','path':str(first),'sha256':sha(first)},
        {'record_id':'r1','path':str(second),'sha256':sha(second)}], [], tmp_path/'out',
        model_path=model, model_sha256=sha(model), feature_model_path='unused', target=1,
        frames_per_parent=2, relax_fn=relax, encoder_factory=lambda _:None,
        extract_fn=extract, sample_fn=sample)
    assert events == ['relax','relax','perturb','perturb','extract','direct']


def test_cli_resolves_bundle_inputs_and_binds_manifest(tmp_path, monkeypatch):
    import phase_agent.science.mlip.parent_first_sampling as worker
    root=tmp_path/'input'; root.mkdir()
    parent=root/'p.vasp'; parent.write_text('parent',encoding='utf8')
    reference=root/'r.vasp'; reference.write_text('reference',encoding='utf8')
    manifest=root/'input_manifest.json'
    data={'schema':'parent-first-hpc-input-v1','scope_id':'scope','round_evidence_digest':'digest',
          'baseline_hull_version':'hull','parents':[{'path':'p.vasp','sha256':sha(parent)}],
          'references':[{'path':'r.vasp','sha256':sha(reference)}], 'model_path':'/model',
          'model_sha256':'a'*64,'feature_model_path':'/feature','feature_model_sha256':'b'*64,
          'target':100,'frames_per_parent':36,'seed':42,'min_distance_A':1.0,'volume_ratio':[0.75,1.25]}
    manifest.write_text(json.dumps(data),encoding='utf8')
    seen={}
    def fake(parents,references,output,**kwargs):
        seen.update(parents=parents,references=references,kwargs=kwargs)
        return {'status':'completed'}
    monkeypatch.setattr(worker,'run_parent_first_sampling',fake)
    worker.main(['--input',str(manifest),'--output',str(tmp_path/'results'/'selected'),
                 '--checkpoint-dir',str(tmp_path/'input'/'checkpoints')])
    assert seen['parents'][0]['path']==str(parent.resolve())
    assert seen['kwargs']['binding']['input_manifest_sha256']==sha(manifest)
    with pytest.raises(ValueError,match='checkpoint_inside_results'):
        worker.main(['--input',str(manifest),'--output',str(tmp_path/'results'/'selected'),
                     '--checkpoint-dir',str(tmp_path/'results'/'checkpoints')])


def test_real_direct_pipeline_with_synthetic_feature_encoder(tmp_path):
    import numpy as np
    atoms=Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]], cell=[6,6,6], pbc=True)
    parent=tmp_path/'p.vasp';write(parent,atoms,format='vasp')
    model=tmp_path/'model';model.write_bytes(b'model')
    def relax(job):
        final=Path(job['output_directory'])/'initial_relaxed.vasp';final.parent.mkdir()
        write(final,read(job['structure_path']),format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(final),'structure_checksum':sha(final),
                'extrapolation_monitor':{'extrapolation_failure':False,'failures':[], 'final_feature_difference':0.1}}
    class Encoder:
        metadata={'type':'synthetic','model_sha256':'b'*64}
        def __call__(self, structure):
            return np.concatenate([structure.positions.reshape(-1), structure.cell.array.reshape(-1)])
    result=run_parent_first_sampling([{'record_id':'r','path':str(parent),'sha256':sha(parent)}], [], tmp_path/'out',
        model_path=model,model_sha256=sha(model),feature_model_path='unused',feature_model_sha256='b'*64,
        target=2,frames_per_parent=5,relax_fn=relax,encoder_factory=lambda _:Encoder())
    assert result['status']=='completed' and result['selected_count']==2
