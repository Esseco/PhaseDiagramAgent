"""Agent-owned registration of immutable same-frame prediction evidence."""
import json
from pathlib import Path
from filelock import FileLock


def register(control, input_archive, result_archive):
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    from phase_agent.tools.workflows.same_frame_comparison_import import import_comparisons
    path=Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(path.parent/'langgraph_lifecycle.sqlite')+'.lock',timeout=30):
        before=checksum(path)
        state=json.loads(path.read_text(encoding='utf8'))
        source_hash=checksum(input_archive); returned_hash=checksum(result_archive)
        prior=(state.get('same_frame_comparison_imports') or {}).get(source_hash)
        if prior:
            if prior['result_sha256']!=returned_hash: raise ValueError('comparison_result_conflict')
            return {'status':'already_registered','count':prior['count']}
        updated,receipt=import_comparisons(state,input_archive,result_archive)
        updated.setdefault('same_frame_comparison_imports',{})[source_hash]={
            'input_sha256':source_hash,'result_sha256':returned_hash,'count':receipt['count'],
            'model_version':receipt['model_version']}
        commit_result_recovery(path,updated,{},expected_hashes={str(path):before},
            evidence={'operation':'same_frame_comparison_registration','input_sha256':source_hash,
                      'result_sha256':returned_hash})
        control._codex_context_cache=None
        return receipt
