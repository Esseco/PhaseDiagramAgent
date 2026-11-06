import hashlib
import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from phase_agent.science.mlip.parent_first_sampling import run_parent_first_sampling
from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_hpc_return_binds_inputs_model_and_selected_frames(tmp_path):
    inp=tmp_path/'input'; inp.mkdir()
    atoms=Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]], cell=[6,6,6], pbc=True)
    parent=inp/'p.vasp'; reference=inp/'r.vasp'; write(parent,atoms,format='vasp'); write(reference,atoms,format='vasp')
    model=tmp_path/'model'; model.write_bytes(b'model')
    row={'record_id':'r1','branch_id':'b1','phase':'O3','H':[[1,0,0],[0,2,0],[0,0,1]],
         'x_Na_per_O2':0.5,'path':'p.vasp','sha256':sha(parent)}
    ref={'task_id':'DFT-1','path':'r.vasp','sha256':sha(reference)}
    data={'schema':'parent-first-hpc-input-v1','scope_id':'scope','round_evidence_digest':'digest',
          'baseline_hull_version':'hull','model_sha256':sha(model),'feature_model_sha256':'b'*64,
          'frames_per_parent':2,'target':1,'parents':[row],'references':[ref]}
    manifest=inp/'input_manifest.json';manifest.write_text(json.dumps(data),encoding='utf8')
    binding={k:data[k] for k in ('scope_id','round_evidence_digest','baseline_hull_version')}
    binding['input_manifest_sha256']=sha(manifest)
    paths=[]
    def relax(job):
        final=Path(job['output_directory'])/'initial_relaxed.vasp';final.parent.mkdir()
        write(final,read(job['structure_path']),format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(final),
                'structure_checksum':sha(final),'extrapolation_monitor':{'extrapolation_failure':False,
                'failures':[],'final_feature_difference':0.2}}
    def extract(inputs, encoder, cache):
        paths.extend(inputs);cache.write_text('fake',encoding='utf8')
    def sample(cache, output, target, reference_count):
        assert reference_count==1
        output.mkdir();(output/'selected_indices.json').write_text(json.dumps([{'source':paths[1],'frame':0}]),encoding='utf8')
        return {'n_samples':1}
    class Encoder:
        metadata={'model_sha256':'b'*64}
    output=tmp_path/'result'
    run_parent_first_sampling([{**row,'path':str(parent)}],[{**ref,'path':str(reference)}],output,
        model_path=model,model_sha256=sha(model),feature_model_path='unused',feature_model_sha256='b'*64,
        target=1,frames_per_parent=2,binding=binding,relax_fn=relax,encoder_factory=lambda _:Encoder(),
        extract_fn=extract,sample_fn=sample)
    checked=validate_parent_first_result(inp,output)
    assert checked['status']=='verified' and checked['reference_count']==1 and len(checked['selected'])==1
    (output/'selected_0000.vasp').write_text('tampered',encoding='utf8')
    with pytest.raises(ValueError,match='selected_binding_invalid'):
        validate_parent_first_result(inp,output)
