import json
import subprocess
import sys
import time

import pytest

from phase_agent.tools.step_runner.file_protocol import read_json, write_json, write_json_if_version
from phase_agent.persistence.state.read_versions import read_identity, verify_read


def test_state_read_rejects_cross_process_replacement(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    assert read_json(path)["value"] == 1
    identity = read_identity(path)
    code = (
        "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); "
        "q=p.with_suffix('.new'); q.write_text('{\"value\": 2}'); os.replace(q,p)"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    with pytest.raises(ValueError, match="state_changed_during_read"):
        verify_read(path, identity)


def test_state_read_detects_control_revision_change(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    assert read_json(path)["value"] == 1
    control = path.with_name("state.json.control.json")
    control.write_text(
        json.dumps(
            {"state_path": str(path.resolve()), "revision": "other", "binding": {}, "patch": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale_(state_file_version|workflow_control_revision)"):
        write_json(path, {"value": 2})


def test_stale_read_cannot_overwrite_another_process(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    old = read_json(path)
    version = read_identity(path)
    code = (
        "import json,sys; from pathlib import Path; "
        "from phase_agent.tools.step_runner.file_protocol import read_json,write_json; "
        "p=Path(sys.argv[1]); s=read_json(p); s['value']=2; write_json(p,s)"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    old["other"] = "stale"
    with pytest.raises(ValueError, match="stale_state_file_version"):
        write_json_if_version(path, old, version)
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}


def test_concurrent_readers_cannot_both_overwrite_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"value": 0}', encoding="utf-8")
    script = """
import sys, time
from pathlib import Path
from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version
from phase_agent.persistence.state.read_versions import read_identity
p, signal, ready, number = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
state = read_json(p)
version = read_identity(p)
ready.touch()
deadline = time.monotonic() + 15
while not signal.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('barrier timeout')
    time.sleep(0.01)
state['value'] = number
try:
    write_json_if_version(p, state, version)
except ValueError as error:
    if 'stale_state_file_version' not in str(error):
        raise
    print('stale')
else:
    print('saved')
"""
    signal = tmp_path / "go"
    ready = [tmp_path / "ready1", tmp_path / "ready2"]
    processes = [
        subprocess.Popen([sys.executable, "-c", script, str(path), str(signal), str(ready[i]), str(i + 1)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for i in range(2)
    ]
    try:
        deadline = time.monotonic() + 15
        while not all(p.exists() for p in ready):
            assert time.monotonic() < deadline, "reader barrier timed out"
            time.sleep(0.01)
        signal.touch()
        outcomes = [p.communicate(timeout=20) for p in processes]
        assert all(p.returncode == 0 for p in processes), outcomes
        assert sorted(out.strip() for out, _ in outcomes) == ["saved", "stale"]
        assert json.loads(path.read_text(encoding="utf-8"))["value"] in {1, 2}
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
