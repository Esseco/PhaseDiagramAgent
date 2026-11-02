import hashlib
from ase import Atoms
from ase.io import write
from phase_agent.analysis.phase.latest_mlip_lineage import latest_mlip_records

def make(tmp_path,name,positions):
    p=tmp_path/name
    write(p,Atoms('NaO',positions=positions,cell=[5,5,5],pbc=True),format='vasp')
    return str(p),hashlib.sha256(p.read_bytes()).hexdigest()

def row(path,sha,version,rid):
    return {'record_id':rid,'structure_id':'S','structure_sha256':sha,'structure_path':path,'energy_method':'mlip','model_version':version,'energy':-2,'status':'completed'}

def test_identical_geometry_latest_epoch_but_different_geometry_preserved(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    b,hb=make(tmp_path,'b.vasp',[[5,0,0],[1,1,1]])
    c,hc=make(tmp_path,'c.vasp',[[0,0,0],[2,1,1]])
    selected,issues=latest_mlip_records([row(a,ha,'m0','a'),row(b,hb,'m1','b'),row(c,hc,'m0','c')],[],{'m0':1,'m1':2})
    assert {r['record_id'] for r in selected}=={'b','c'} and not issues

def test_relaxation_lineage_and_same_epoch_ambiguity(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    b,hb=make(tmp_path,'b.vasp',[[0,0,0],[1.2,1,1]])
    old=row(a,ha,'m0','a'); new={**row(b,hb,'m1','b'),'source_task_id':'t'}
    task={'task_id':'t','structure_id':'S','model_version':'m1','status':'completed','input_structure_sha256':ha,'parameters':{'model_refresh_operation':'relax'}}
    selected,issues=latest_mlip_records([old,new],[task],{'m0':1,'m1':2})
    assert [r['record_id'] for r in selected]==['b'] and not issues
    another={**new,'record_id':'other','energy':-3}
    selected,issues=latest_mlip_records([old,new,another],[task],{'m0':1,'m1':2})
    assert not selected and issues[0]['reason']=='ambiguous_same_epoch_latest'

def test_failed_latest_does_not_replace_valid_old(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    old=row(a,ha,'m0','a'); failed={**row(a,ha,'m1','b'),'checks_passed':False}
    selected,issues=latest_mlip_records([old,failed],[],{'m0':1,'m1':2})
    assert [r['record_id'] for r in selected]==['a'] and not issues

def test_unknown_epoch_is_reported_without_assuming_order(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    selected,issues=latest_mlip_records([row(a,ha,'unknown','a')],[],{'m0':1})
    assert not selected and issues==[{'record_id':'a','reason':'missing_identity_or_model_epoch'}]

def test_two_relaxation_descendants_remain_separate(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    b,hb=make(tmp_path,'b.vasp',[[0,0,0],[1.2,1,1]])
    c,hc=make(tmp_path,'c.vasp',[[0,0,0],[1.5,1,1]])
    old=row(a,ha,'m0','a'); r1={**row(b,hb,'m1','b'),'source_task_id':'t1'};r2={**row(c,hc,'m1','c'),'source_task_id':'t2'}
    tasks=[{'task_id':tid,'structure_id':'S','model_version':'m1','status':'completed','input_structure_sha256':ha,'parameters':{'model_refresh_operation':'relax'}} for tid in ['t1','t2']]
    selected,issues=latest_mlip_records([old,r1,r2],tasks,{'m0':1,'m1':2})
    assert {r['record_id'] for r in selected}=={'b','c'} and not issues

def test_reordered_same_species_are_same_geometry(tmp_path):
    a=tmp_path/'a.vasp';b=tmp_path/'b.vasp'
    x=Atoms('NaNaO',positions=[[0,0,0],[1,1,1],[2,2,2]],cell=[5,5,5],pbc=True)
    write(a,x,format='vasp');write(b,x[[1,0,2]],format='vasp')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    selected,issues=latest_mlip_records([row(str(a),sha(a),'m0','a'),row(str(b),sha(b),'m1','b')],[],{'m0':1,'m1':2})
    assert [r['record_id'] for r in selected]==['b'] and not issues

def test_same_epoch_uses_bound_task_completion_time(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    x={**row(a,ha,'m1','x'),'source_task_id':'t1','energy':-2}
    y={**row(a,ha,'m1','y'),'source_task_id':'t2','energy':-3}
    tasks=[{'task_id':'t1','structure_id':'S','model_version':'m1','status':'completed','runtime_observation':{'ended_at_epoch':10}}, {'task_id':'t2','structure_id':'S','model_version':'m1','status':'completed','runtime_observation':{'ended_at_epoch':20}}]
    selected,issues=latest_mlip_records([x,y],tasks,{'m1':2})
    assert [r['record_id'] for r in selected]==['y'] and not issues
    selected,issues,aliases=latest_mlip_records([x,y],tasks,{'m1':2},return_aliases=True)
    assert aliases=={'x':'y'}

def test_duplicate_task_identity_is_rejected(tmp_path):
    import pytest
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    with pytest.raises(ValueError,match='duplicate_task_id'):
        latest_mlip_records([row(a,ha,'m0','a')],[{'task_id':'t'},{'task_id':'t'}],{'m0':1})

def test_model_epoch_collision_is_rejected(tmp_path):
    import pytest
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    with pytest.raises(ValueError,match='duplicate_model_epoch'):
        latest_mlip_records([row(a,ha,'m0','a')],[],{'m0':1,'m1':1})

def test_missing_record_identity_is_reported(tmp_path):
    a,ha=make(tmp_path,'a.vasp',[[0,0,0],[1,1,1]])
    selected,issues=latest_mlip_records([row(a,ha,'m0',None)],[],{'m0':1})
    assert not selected and issues[0]['reason']=='missing_identity_or_model_epoch'

def test_alias_resolution_rejects_cross_structure_and_model():
    import pytest
    from phase_agent.analysis.phase.latest_mlip_lineage import resolve_latest_task_aliases
    source={'record_id':'old','source_task_id':'t','structure_id':'s','source_version':'m1','structure_sha256':'same'}
    target={'record_id':'latest','structure_id':'s','source_version':'m1','structure_sha256':'same'}
    phases,superseded=resolve_latest_task_aliases([target],[source],{'old':'latest'},'m1')
    assert phases['t']==target and superseded=={'t'}
    with pytest.raises(ValueError,match='invalid_latest_record_alias'):
        resolve_latest_task_aliases([{**target,'structure_id':'different'}],[source],{'old':'latest'},'m1')
    with pytest.raises(ValueError,match='invalid_latest_record_alias'):
        resolve_latest_task_aliases([{**target,'source_version':'other'}],[source],{'old':'latest'},'m1')
