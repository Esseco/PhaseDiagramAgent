"""Select latest MLIP energy for each evidenced geometry and directed refresh lineage."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from ase.io import read
from scipy.optimize import linear_sum_assignment


def structures_equivalent(a, b, caches=None):
    """Compare same-cell atomic geometries, permitting periodic wrapping and atom order."""
    if not a.get('structure_sha256') or not b.get('structure_sha256'):
        raise ValueError('geometry_checksum_missing')
    if a['structure_sha256'] == b['structure_sha256']:
        return True
    caches = {} if caches is None else caches
    def geometry(record):
        path=record.get('structure_path')
        if not path or not Path(path).is_file():raise ValueError('geometry_source_missing')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=record['structure_sha256']:
            raise ValueError('geometry_checksum_mismatch')
        if path not in caches:caches[path]=read(path,format='vasp')
        return caches[path]
    x,y=geometry(a),geometry(b)
    if sorted(x.get_chemical_symbols())!=sorted(y.get_chemical_symbols()) or not np.allclose(x.cell.array,y.cell.array,atol=1e-5,rtol=0):return False
    sx,sy=x.get_scaled_positions(wrap=False),y.get_scaled_positions(wrap=False)
    xnums,ynums=x.get_atomic_numbers(),y.get_atomic_numbers()
    for element in set(xnums):
        a,b=sx[xnums==element],sy[ynums==element]
        delta=a[:,None,:]-b[None,:,:];delta-=np.rint(delta)
        distances=np.linalg.norm(delta @ x.cell.array,axis=2)
        i,j=linear_sum_assignment(distances)
        if np.any(distances[i,j]>1e-5):return False
    return True


def latest_mlip_records(records, tasks, model_rounds, *, return_aliases=False):
    if len(set(model_rounds.values())) != len(model_rounds):
        raise ValueError('duplicate_model_epoch')
    bytask = {}
    for task in tasks:
        task_id=task.get('task_id')
        if not task_id:continue
        if task_id in bytask:raise ValueError('duplicate_task_id')
        bytask[task_id]=task
    parent, groups, caches, issues = {}, defaultdict(list), {}, []

    def find(k):
        parent.setdefault(k, k)
        if parent[k] != k:
            parent[k] = find(parent[k])
        return parent[k]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    def equivalent(a, b):
        return structures_equivalent(a,b,caches)

    eligible=[]
    for r in records:
        if r.get('energy_method') != 'mlip' or r.get('status','completed') != 'completed' or r.get('checks_passed',True) is not True:
            continue
        sid, sha = r.get('structure_id'), r.get('structure_sha256')
        version = r.get('model_version') or r.get('source_version')
        epoch=model_rounds.get(version)
        if not r.get('record_id') or not sid or not sha or type(epoch) is not int or epoch < 1:
            issues.append({'record_id':r.get('record_id'),'reason':'missing_identity_or_model_epoch'})
            continue
        find((sid,sha));groups[sid].append(r);eligible.append(r)

    for sid, rows in groups.items():
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if find((sid,a['structure_sha256'])) != find((sid,b['structure_sha256'])) and equivalent(a,b):
                    union((sid,a['structure_sha256']),(sid,b['structure_sha256']))

    # Directed edges preserve branches: one parent can generate multiple relaxed geometries.
    edges=defaultdict(set)
    for r in eligible:
        sid, version = r['structure_id'], r.get('model_version') or r.get('source_version')
        task = bytask.get(r.get('source_task_id'))
        operation=(task.get('parameters') or {}).get('model_refresh_operation') if task else None
        if task and task.get('status') == 'completed' and task.get('checks_passed',True) is True and operation in {'predict','relax'} and task.get('structure_id') == sid and task.get('model_version') == version and task.get('input_structure_sha256'):
            a=find((sid,task['input_structure_sha256'])); b=find((sid,r['structure_sha256']))
            if a!=b: edges[a].add(b)
    # A cycle would make "latest" undefined; never silently drop every member.
    visiting,visited=set(),set()
    def check_cycle(node):
        if node in visiting: raise ValueError('cyclic_refresh_lineage')
        if node in visited:return
        visiting.add(node)
        for child in edges.get(node,()):check_cycle(child)
        visiting.remove(node);visited.add(node)
    for node in list(edges):check_cycle(node)
    grouped=defaultdict(list)
    for r in eligible: grouped[find((r['structure_id'],r['structure_sha256']))].append(r)
    winners={}
    for key, rows in grouped.items():
        top=max(model_rounds[r.get('model_version') or r.get('source_version')] for r in rows)
        latest=[r for r in rows if model_rounds[r.get('model_version') or r.get('source_version')]==top]
        if len(latest)>1:
            times=[]
            for record in latest:
                task=bytask.get(record.get('source_task_id')) or {}
                value=(task.get('runtime_observation') or {}).get('ended_at_epoch')
                if task.get('status')!='completed' or task.get('structure_id')!=record.get('structure_id') or task.get('model_version')!=(record.get('model_version') or record.get('source_version')) or not isinstance(value,(int,float)) or isinstance(value,bool) or not np.isfinite(value) or value<=0:
                    times=[];break
                times.append(value)
            if times and times.count(max(times))==1:
                latest=[latest[times.index(max(times))]]
        energies=[r.get('energy') for r in latest]
        if len(latest)>1 and not (all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in energies) and max(energies)-min(energies)<=1e-8 and all(equivalent(latest[0],r) for r in latest[1:])):
            issues.append({'structure_id':key[0],'reason':'ambiguous_same_epoch_latest','record_ids':[r.get('record_id') for r in latest]})
            continue
        winners[key]=sorted(latest,key=lambda r:str(r.get('record_id')))[-1]

    def has_newer_descendant(key, baseline):
        seen={key}; stack=list(edges.get(key,()))
        while stack:
            current=stack.pop()
            if current in seen:
                continue
            seen.add(current)
            descendants=grouped.get(current,[])
            if descendants and max(model_rounds[r.get('model_version') or r.get('source_version')] for r in descendants) >= baseline:
                return True
            stack.extend(edges.get(current,()))
        return False

    selected=[r for key,r in winners.items() if not has_newer_descendant(key,model_rounds[r.get('model_version') or r.get('source_version')])]
    if not return_aliases:
        return selected,issues
    selected_ids={r.get('record_id') for r in selected}
    aliases={}
    for key,winner in winners.items():
        if winner.get('record_id') not in selected_ids:continue
        version=winner.get('model_version') or winner.get('source_version')
        for record in grouped[key]:
            rid=record.get('record_id')
            if rid and rid!=winner.get('record_id') and (record.get('model_version') or record.get('source_version'))==version:
                aliases[rid]=winner.get('record_id')
    return selected,issues,aliases


def resolve_latest_task_aliases(entries, records, aliases, model_version):
    """Resolve only same-model, same-structure aliases from the frozen mixed hull."""
    by_record={r.get('record_id'):r for r in records if r.get('record_id')}
    entry_by_id={r.get('record_id'):r for r in entries if r.get('record_id')}
    phases, superseded = {}, set()
    for source_id, target_id in aliases.items():
        source=by_record.get(source_id); target=entry_by_id.get(target_id)
        if not source or not target or source.get('source_version')!=model_version or target.get('source_version')!=model_version or not source.get('source_task_id') or source.get('structure_id')!=target.get('structure_id') or not structures_equivalent(source,target):
            raise ValueError('invalid_latest_record_alias')
        phases[source['source_task_id']]=target
        superseded.add(source['source_task_id'])
    return phases,superseded
