"""Agent entry for verified externally-run diagnostic DFT retry results."""
from pathlib import Path
import json
from filelock import FileLock


def register(control, input_archive, result_archive):
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.tools.workflows.diagnostic_retry_import import register_diagnostic_retry
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery

    state_path=Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(state_path.parent/'langgraph_lifecycle.sqlite')+'.lock',timeout=30):
        before=checksum(state_path)
        state=json.loads(state_path.read_text(encoding='utf8'))
        batch_list=state.get('parent_first_recovery_batches') or {}
        if len(batch_list)!=1: raise ValueError('diagnostic_retry_project_batch_ambiguous')
        batch=next(iter(batch_list.values()))
        original=Path(batch['input_dir']).resolve()
        epoch=next((p for p in original.parents if p.name.startswith('epoch') and p.name[5:].isdigit()),None)
        if epoch is None: raise ValueError('diagnostic_retry_epoch_missing')
        evidence=epoch/'evidence'/'verified_results'/('diagnostic-retry-'+checksum(input_archive)[:16])
        result=register_diagnostic_retry(state,input_archive,result_archive,evidence)
        if result['status']=='registered':
            commit_result_recovery(state_path,result['state'],{},expected_hashes={str(state_path):before},
                evidence={'operation':'diagnostic_dft_retry_registration','input_sha256':result['input_sha256'],
                          'result_sha256':result['result_sha256']})
        control._codex_context_cache=None
        return {k:v for k,v in result.items() if k!='state'}
