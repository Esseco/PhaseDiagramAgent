import zipfile
from phase_agent.tools.remote.handoff_naming import package_folder


def test_malformed_prior_package_is_preserved_and_new_sequence_is_used(tmp_path):
    first=tmp_path/'DFTSP_Batch01_output'
    first.mkdir()
    prior=tmp_path/'DFTSP_Batch02_input'
    prior.mkdir()
    archive=prior/'inputs.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('wrong.json','{}')
    before=archive.read_bytes()
    assert package_folder(tmp_path,'DFTSP','abcdef')==tmp_path/'DFTSP_Batch03_input'
    assert archive.read_bytes()==before
