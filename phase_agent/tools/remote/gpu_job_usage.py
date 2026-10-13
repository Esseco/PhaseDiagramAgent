"""Standalone GPU allocation clock, also usable to recover known Slurm jobs."""
import json,os,re,time,sys,subprocess
from pathlib import Path

def number(value):
    try:
        result=float(value)
        return result if result>=0 and result<float('inf') else None
    except (TypeError,ValueError): return None

def gpu_count(env):
    value=number(env.get('SLURM_GPUS'))
    if value is not None: return value
    value=env.get('SLURM_GPUS_ON_NODE')
    count=number(value)
    if count is None and value:
        match=re.fullmatch(r'(?:gpu:)?[A-Za-z][\w.-]*:(\d+)',str(value))
        count=number(match.group(1)) if match else None
    if count is not None: return count*(number(env.get('SLURM_JOB_NUM_NODES')) or 1)
    ids=env.get('SLURM_JOB_GPUS')
    if ids and (number(env.get('SLURM_JOB_NUM_NODES')) or 1)==1:
        total=0
        for part in ids.split(','):
            match=re.fullmatch(r'(\d+)(?:-(\d+))?',part.strip())
            if not match:return None
            total+=int(match.group(2))-int(match.group(1))+1 if match.group(2) else 1
        return float(total)
    return None

def result_root(directory):
    directory=Path(directory).resolve()
    for parent in (directory,*directory.parents):
        if parent.name=='inputs': return parent.parent/'results'
    return directory/'results'

def stage(directory):
    directory=Path(directory)
    if (directory/'training_plan.json').is_file():return 'mlip_training'
    try:
        task=json.loads((directory/'task.json').read_text())
        return task.get('stage','unknown')
    except (OSError,ValueError):pass
    try:
        rows=json.loads((directory/'manifest.json').read_text())
        return rows[0].get('stage','unknown') if rows else 'unknown'
    except (OSError,ValueError):return 'unknown'

def save(root,row):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=re.sub(r'[^A-Za-z0-9_-]','_',row['job_id'])
    attempt=str(row.get('restart_count',0))
    path=root/f'gpu-job-{name}-{attempt}.runtime.json'
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(temp,path)
    # Rebuild the normal checksummed ZIP after the job clock is written.
    try:
        from _gpu_transfer_bundle import consolidate_results
    except ImportError:
        from phase_agent.tools.remote.transfer_bundle import consolidate_results
    consolidate_results(root)
    return path

def finish(start,exit_code,directory=None,env=None):
    env=os.environ if env is None else env
    if not env.get('SLURM_JOB_ID'):return None
    directory=Path(directory or env.get('SLURM_SUBMIT_DIR') or Path.cwd())
    elapsed=max(0,time.time()-float(start));gpus=gpu_count(env)
    return save(result_root(directory),{'schema':1,'source':'gpu_job_clock','allocation_basis':'whole_gpu_job','job_id':env['SLURM_JOB_ID'],'cluster':env.get('SLURM_CLUSTER_NAME',''),'restart_count':env.get('SLURM_RESTART_COUNT','0'),'elapsed_seconds':elapsed,'gpu_count':gpus,'actual_gpu_core_hours':None if gpus is None else elapsed*gpus/3600,'stage':stage(directory),'exit_code':int(exit_code),'started_at_epoch':float(start),'ended_at_epoch':time.time()})

def collect(directory):
    directory=Path(directory)
    ids=set()
    for path in directory.rglob('*.json'):
        if path.name not in {'result.json','.submitted_jobs.json'} and not path.name.endswith('.runtime.json'):continue
        try:value=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        def walk(row):
            if isinstance(row,dict):
                for key,item in row.items():
                    if key in {'job_id','slurm_job_id'} and re.fullmatch(r'\d+(?:_\d+)?',str(item)):ids.add(str(item))
                    else:walk(item)
            elif isinstance(row,list):
                for item in row:walk(item)
        walk(value)
    if not ids:raise ValueError('No known job IDs found in result/receipt files')
    result=subprocess.run(['sacct','-n','-P','-X','-j',','.join(sorted(ids)),'--format=JobIDRaw,ElapsedRaw,AllocTRES,State,Cluster'],check=True,capture_output=True,text=True)
    count=0
    for line in result.stdout.splitlines():
        cells=line.split('|')
        if len(cells)<5 or cells[0] not in ids or cells[3] in {'PENDING','RUNNING','CONFIGURING','COMPLETING'}:continue
        seconds=number(cells[1]);matches=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',cells[2])
        if not matches:
            typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',cells[2])
            if typed:matches=[str(sum(int(n) for n in typed))]
        if seconds is None or not matches:continue
        gpus=float(matches[0]);row={'schema':1,'source':'sacct','allocation_basis':'whole_gpu_job','job_id':cells[0],'cluster':cells[4],'restart_count':'0','elapsed_seconds':seconds,'gpu_count':gpus,'actual_gpu_core_hours':seconds*gpus/3600,'stage':'scheduler_recovered','job_state':cells[3]}
        save(result_root(directory),row);count+=1
    return count

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--collect':print(collect(sys.argv[2]))
    else:finish(sys.argv[1],sys.argv[2])
