import hashlib
import json
import zipfile
import pytest
from phase_agent.tools.remote.transfer_bundle import import_bundle, MANIFEST, MAX_MANIFEST_BYTES
from phase_agent.runtime.waiting_files_presentation import waiting_file_lines
from phase_agent.runtime.studio_reply_presentation import format_turn_reply


def archive(tmp_path, padding):
    path=tmp_path/'results.zip'
    manifest={'schema':1,'kind':'results','files':[{'path':'result.json','size':2,'sha256':hashlib.sha256(b'{}').hexdigest()}]}
    raw=json.dumps(manifest).encode()+b' '*padding
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST,raw); z.writestr('result.json',b'{}')
    return path


def test_legitimate_manifest_over_four_mib(tmp_path):
    assert import_bundle(archive(tmp_path,4*1024**2),tmp_path/'out')==1


def test_manifest_still_bounded(tmp_path):
    with pytest.raises(ValueError,match='manifest exceeds size limit'):
        import_bundle(archive(tmp_path,MAX_MANIFEST_BYTES),tmp_path/'out')
    assert not (tmp_path/'out/result.json').exists()


def test_rejected_return_never_prompts_resubmission(tmp_path):
    state={'tasks':[{'status':'pending','stage':'relax_and_feature','relax_mc_bundle_id':'b'}], 'relax_mc_batches':{'b':{'directory':str(tmp_path/'epoch0_RelaxMC_Batch01'),'return_error':'Transfer manifest exceeds size limit'}}}
    before=json.dumps(state,sort_keys=True)
    text='\n'.join(waiting_file_lines(state))
    reply=format_turn_reply('尚无回收，请上传 inputs 包并执行 sbatch submit_all.sh',state)
    for value in [text,reply]:
        assert '无需重新上传输入或提交计算' in value
        assert 'sbatch' not in value
    assert json.dumps(state,sort_keys=True)==before
