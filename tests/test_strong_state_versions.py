"""Strong state versions detect same-metadata changes without copying large files."""

import subprocess
import sys

import pytest

from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json, write_json, write_json_if_version


def _other_process_same_metadata(path):
    before = path.stat()
    code = (
        "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); st=p.stat(); "
        "b=p.read_bytes(); old=b'\\\"A\\\"'; new=b'\\\"B\\\"'; "
        "assert old in b; p.write_bytes(b.replace(old,new,1)); os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino
    )


def test_strong_state_version_rejects_same_metadata_change(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"audit": "A"})
    version = read_identity(path, strong=True)
    _other_process_same_metadata(path)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        write_json_if_version(path, {"audit": "A", "local": True}, version)
    assert read_json(path) == {"audit": "B"}


def test_strong_identity_detects_same_metadata_control_change(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"audit": "keep"})
    control = path.with_name(path.name + ".control.json")
    control.write_text('{"audit":"A"}', encoding="utf8")
    version = read_identity(path, strong=True)
    _other_process_same_metadata(control)
    assert read_identity(path, strong=True) != version


def test_legacy_weak_identity_remains_readable(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"audit": "A"})
    version = read_identity(path)
    write_json_if_version(path, {"audit": "B"}, version)
    assert read_json(path) == {"audit": "B"}
