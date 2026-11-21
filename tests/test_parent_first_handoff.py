import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_parent_first_bundle_only_contains_parent_reference_and_parameters(tmp_path, monkeypatch):
    import phase_agent.agents.reliability.extrapolation_round as reliability
    import phase_agent.evidence.perturbation_recovery as recovery
    import phase_agent.tools.workflows.comparison_model_registry as registry
    import phase_agent.tools.workflows.direct_extrapolation_sampling as direct
    from phase_agent.tools.workflows.parent_first_handoff import prepare_parent_first_bundle
    atoms = Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]],
                  cell=[6,6,6], pbc=True)
    source = tmp_path/'parent.vasp'; write(source, atoms, format='vasp')
    parent = {'record_id':'r','branch_id':'b','phase':'O3','H':[[1,0,0],[0,2,0],[0,0,1]],
              'x_Na_per_O2':0.5,'structure_path':str(source),'structure_sha256':sha(source)}
    monkeypatch.setattr(reliability, 'required_direct_round', lambda state: {'scope_id':'scope','evidence_digest':'digest'})
    monkeypatch.setattr(recovery, 'build_perturbation_recovery_pool', lambda *a, **k: {'hull_version':'old', 'candidates':[parent]})
    monkeypatch.setattr(registry, 'original_round_model', lambda *a: ({'framework':'mace','mace_head':'omat_pbe','model_path':'/remote/mh1.model'}, 'a'*64, None))
    monkeypatch.setattr(direct, 'successful_dft_references', lambda state: [{'task_id':'DFT-1','structure':AseAtomsAdaptor.get_structure(atoms).as_dict()}])
    params = {'confirmed':True, 'scope_id':'scope', 'round_evidence_digest':'digest',
              'direct_target':100, 'max_det_H':3, 'parent_record_ids':['r']}
    result = prepare_parent_first_bundle({}, {}, params, tmp_path/'bundle',
        feature_model_path='/remote/m3gnet', feature_model_sha256='b'*64)
    assert result['status'] == 'prepared' and result['receipt']['parent_count'] == 1
    assert result['receipt']['dft_reference_count'] == 1 and result['receipt']['target'] == 100
    assert sorted(p.name for p in (tmp_path/'bundle').iterdir()) == ['inputs.zip','receipt.json']
    with zipfile.ZipFile(result['archive']) as archive:
        assert set(archive.namelist()) == {'parents/0000.vasp','references/0000.vasp','input_manifest.json','transfer_manifest.json'}
        payload = json.loads(archive.read('input_manifest.json'))
        assert payload['parents'][0]['record_id'] == 'r'
        assert payload['references'][0]['task_id'] == 'DFT-1'
        assert payload['target'] == 100
    source.write_text('changed', encoding='utf8')
    with pytest.raises(ValueError, match='parent_changed'):
        prepare_parent_first_bundle({}, {}, params, tmp_path/'bad',
            feature_model_path='/remote/m3gnet', feature_model_sha256='b'*64)
    assert not (tmp_path/'bad').exists()
    assert not list(tmp_path.glob('.parent-first-input-*'))
