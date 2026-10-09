"""Local append-only lifecycle timings, without raw prompts or scientific data."""
from pathlib import Path
import json
import time
from filelock import FileLock


def append_node_trace(frame,node,elapsed,*,error=None):
    path=frame.get('state_path') or (frame.get('effective_config') or {}).get('state_path')
    if not path:return
    output=Path(path).resolve().parent/'node_trace.jsonl'
    output.parent.mkdir(parents=True,exist_ok=True)
    record={'timestamp_unix':time.time(),'node':node,'elapsed_seconds':round(elapsed,6),
        'status':'failed' if error else frame.get('status','completed'),
        'error_type':type(error).__name__ if error else None}
    # Only actual reported use is carried; absent use is unknown, not zero.
    record['llm_usage']=frame.get('_llm_usage')
    wait=frame.get('manual_wait') or {}
    record['waiting_by_stage']=wait.get('waiting_by_stage')
    record['recovered_count']=frame.get('recovered_count')
    with FileLock(str(output)+'.lock',timeout=10):
        with output.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
