import json
import zipfile
from pathlib import Path
from unittest.mock import patch
import pytest
from phase_agent.tools.remote import transfer_limits as limits
from phase_agent.tools.remote.transfer_bundle import create_bundle, import_bundle, validate_bundle


def test_capacity_boundaries():
    limits.validate_capacity(limits.MAX_ARCHIVE_FILES, limits.MAX_MANIFEST_BYTES, limits.MAX_ARCHIVE_BYTES)
    for args in [(limits.MAX_ARCHIVE_FILES+1,0,0),(1,limits.MAX_MANIFEST_BYTES+1,0),(1,0,limits.MAX_ARCHIVE_BYTES+1)]:
        with pytest.raises(ValueError): limits.validate_capacity(*args)


def test_producer_refuses_incompatible_package_before_publish(tmp_path, monkeypatch):
    source=tmp_path/'result.json';source.write_text('{}')
    dest=tmp_path/'results.zip';dest.write_bytes(b'prior')
    monkeypatch.setattr(limits,'MAX_ARCHIVE_FILES',1)
    with pytest.raises(ValueError,match='file count'): create_bundle(tmp_path,[source],dest,kind='results')
    assert dest.read_bytes()==b'prior'
    assert not Path(str(dest)+'.part').exists()


def test_stream_validator_detects_tampering_without_extraction(tmp_path):
    source=tmp_path/'result.json';source.write_text('{}')
    archive=create_bundle(tmp_path,[source],tmp_path/'results.zip',kind='results')
    assert validate_bundle(archive)==1
    with zipfile.ZipFile(archive) as z: manifest=z.read('transfer_manifest.json')
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('transfer_manifest.json',manifest);z.writestr('result.json','[]')
    with pytest.raises(ValueError,match='checksum'):validate_bundle(archive)
    with pytest.raises(ValueError,match='checksum'):import_bundle(archive,tmp_path/'out')


def test_generated_standalone_bootstrap_uses_shared_limits():
    from phase_agent.tools.relax_mc_batch.handoff import BOOTSTRAP
    namespace={'__name__':'test_bootstrap'}
    exec(compile(BOOTSTRAP,'run_upload.py','exec'),namespace)
    for name in ['MAX_ARCHIVE_FILES','MAX_MANIFEST_BYTES','MAX_ARCHIVE_BYTES']:
        assert namespace[name]==getattr(limits,name)


def test_inspection_does_not_extract(tmp_path):
    from phase_agent.tools.relax_mc_batch.results import inspect_archive
    (tmp_path/'result_summary.json').write_text('{"status":"completed"}')
    (tmp_path/'candidate_pool.json').write_text('{"branch_candidates":[]}')
    archive=create_bundle(tmp_path,list(tmp_path.glob('*.json')),tmp_path/'results.zip',kind='results')
    with patch('tempfile.TemporaryDirectory',side_effect=AssertionError('must not extract')):
        assert inspect_archive(archive)['summary']['status']=='completed'
