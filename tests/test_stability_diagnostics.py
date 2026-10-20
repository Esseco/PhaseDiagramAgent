from copy import deepcopy
import errno
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from filelock import Timeout

from phase_agent.runtime.agent_api import create_server, _approval_page
from phase_agent.runtime.boundary_diagnostics import describe_failure, record_failure
from phase_agent.runtime.project_diagnostics import inspect_project
from phase_agent.runtime.ui_operations import UIOperations


@pytest.mark.parametrize(
    "error,code",
    [
        (PermissionError(), "permission_denied"),
        (OSError(errno.ENOSPC, "full"), "disk_full"),
        (Timeout("lock"), "project_busy"),
        (TimeoutError(), "dependency_unavailable"),
        (ConnectionError(), "dependency_unavailable"),
        (FileNotFoundError(), "missing_file"),
        (RuntimeError("SECRET_TOKEN"), "internal_error"),
    ],
)
def test_typed_failure_has_safe_scope_and_no_automatic_replay(error, code):
    fact = describe_failure(error, boundary="control_operation", object_id="new-task")
    assert fact["code"] == code and fact["execution_effect"] == "unknown"
    assert not fact["automatic_retry"] and not fact["retry_read_only"]
    assert "SECRET_TOKEN" not in json.dumps(fact)


def test_diagnostic_write_failure_preserves_original_fault(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise PermissionError("diagnostic disk blocked")

    monkeypatch.setattr("phase_agent.tools.step_runner.file_protocol._write_json", fail)
    fact = record_failure(tmp_path / "state.json", TimeoutError(), boundary="control_operation")
    assert fact["code"] == "dependency_unavailable" and not fact["evidence_saved"]


def test_bad_scientific_state_does_not_hide_transactions_operations_or_cli(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{broken")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = inspect_project(path)
    assert not report["sections"]["state"]["available"]
    assert report["sections"]["transaction"]["available"]
    assert report["sections"]["operations"]["available"]
    assert report["sections"]["state"]["diagnostic"]["code"] == "invalid_json"
    result = subprocess.run(
        [sys.executable, "-m", "phase_agent.runtime.project_diagnostics", "--state", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0 and json.loads(result.stdout)["read_only"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("bad", ["charts", "status", "pending", "recovery_transaction"])
def test_http_read_fault_isolated_with_evidence_and_other_endpoints_survive(tmp_path, bad):
    store = UIOperations(tmp_path / "state.json", threading.RLock())

    def fail():
        raise OSError(errno.ENOSPC, "SECRET_TOKEN")

    control = SimpleNamespace(
        state_path=tmp_path / "state.json",
        operations=store,
        charts=lambda: {"charts": {}},
        status=lambda: {"ok": True},
        pending=lambda: {"pending": []},
        recovery_transaction=lambda: {"status": "none"},
        diagnostics=lambda: inspect_project(tmp_path / "state.json"),
    )
    setattr(control, bad, fail)
    server = create_server(
        lambda *a, **k: None,
        api_key="a" * 24,
        control_api_key="c" * 24,
        local_control=control,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=3)
    headers = {"Authorization": "Bearer " + "c" * 24}
    route = "recovery/transaction" if bad == "recovery_transaction" else bad
    try:
        connection.request("GET", "/phase/" + route, headers=headers)
        response = connection.getresponse()
        fact = json.loads(response.read())
        assert response.status == 503 and fact["diagnostic"]["code"] == "disk_full"
        assert "SECRET_TOKEN" not in json.dumps(fact)
        connection.request("GET", "/phase/diagnostics", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.request("GET", "/phase/operations", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        assert not list(tmp_path.rglob("*.json")), "GET must not write incident evidence"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_post_failure_returns_saved_incident_and_never_replays(tmp_path):
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise TimeoutError("SECRET_TOKEN")

    control = SimpleNamespace(state_path=tmp_path / "state.json", propose=fail)
    server = create_server(
        lambda *a, **k: None,
        api_key="a" * 24,
        control_api_key="c" * 24,
        local_control=control,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(
            "POST",
            "/phase/propose",
            json.dumps({"instruction": "new project"}),
            {"Authorization": "Bearer " + "c" * 24, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        fact = json.loads(response.read())
        assert response.status == 503 and fact["diagnostic"]["evidence_saved"]
        assert len(calls) == 1 and len(list(tmp_path.rglob("*.json"))) == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.mark.parametrize(
    "bad",
    [
        "/phase/charts",
        "/phase/status",
        "/phase/pending",
        "/phase/recovery/transaction",
        "/phase/diagnostics",
    ],
)
def test_approval_javascript_loads_independent_sections_and_disables_unsafe_approval(tmp_path, bad):
    import shutil

    node = shutil.which("node")
    assert node, "Node required to exercise shipped page JavaScript"
    page = _approval_page()
    script = page[page.index("async function loadAll()") : page.index("function renderOperations")]
    # Exercise the shipped function, with a minimal DOM, rather than checking strings.
    program = (
        r"""
const assert=require('node:assert/strict');
const elements={};const calls=[];const rendered=[];const buttons=[{disabled:false}];
const el=id=>elements[id]??=( {textContent:'',replaceChildren(){},appendChild(){}} );
const document={querySelectorAll:()=>buttons,createElement:()=>({appendChild(){}})};
const renderPlans=()=>{buttons[0].disabled=false;rendered.push('pending')};const renderOperations=()=>rendered.push('operations');const renderRecovery=()=>rendered.push('status');const pretty=JSON.stringify;
const bad=process.argv[2];async function api(path){calls.push(path);if(path===bad)throw Error('injected fault');return {'/phase/recovery/transaction':{status:'none'},'/phase/pending':{},'/phase/charts':{charts:{}},'/phase/status':{},'/phase/diagnostics':{sections:{}}}[path]}
"""
        + script
        + r"""
(async()=>{await loadAll();assert.equal(calls.length,5);assert.ok(el('message').textContent.includes(bad));if(bad!='/phase/pending')assert.ok(rendered.includes('pending'));if(bad!='/phase/status')assert.ok(rendered.includes('status'));if(['/phase/status','/phase/recovery/transaction'].includes(bad))assert.ok(buttons[0].disabled);})().catch(e=>{console.error(e);process.exit(1)});
"""
    )
    path = tmp_path / "ui.js"
    path.write_text(program, encoding="utf-8")
    result = subprocess.run([node, str(path), bad], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_scientific_action_failure_has_incident_without_replaying_effect(tmp_path):
    from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action

    calls = []

    def fail(**kwargs):
        calls.append(1)
        raise PermissionError("simulated locked result")

    action = {"tool": "generate_branches", "task_key": "new-input"}
    context = {
        "state_path": str(tmp_path / "state.json"),
        "invocation_id": "new-action",
        "config_version": "c1",
    }
    result = execute_tool_action(
        action, registry={"generate_branches": {"handler": fail}}, context=context
    )
    assert result["status"] == "failed" and result["diagnostic"]["code"] == "permission_denied"
    assert result["diagnostic"]["execution_effect"] == "unknown"
    replay = execute_tool_action(
        action, registry={"generate_branches": {"handler": fail}}, context=context
    )
    assert replay["status"] == "execution_reconciliation_required" and calls == [1]


def test_hung_chart_does_not_delay_independent_shipped_page_sections(tmp_path):
    import shutil

    node = shutil.which("node")
    assert node
    page = _approval_page()
    script = page[page.index("async function loadAll()") : page.index("function renderOperations")]
    program = (
        r"""
const assert=require('node:assert/strict');const elements={};const rendered=[];const buttons=[{disabled:false}];
const el=id=>elements[id]??={textContent:'',replaceChildren(){},appendChild(){}};
const document={querySelectorAll:()=>buttons,createElement:()=>({appendChild(){}})};
const renderPlans=()=>{buttons[0].disabled=false;rendered.push('pending')};const renderOperations=()=>{};const renderRecovery=()=>rendered.push('status');const pretty=JSON.stringify;
async function api(path){if(path==='/phase/charts')return new Promise(()=>{});return {'/phase/recovery/transaction':{status:'none'},'/phase/pending':{},'/phase/status':{summary_id:'current'},'/phase/diagnostics':{sections:{}}}[path]}
"""
        + script
        + r"""
loadAll();setTimeout(()=>{try{assert.ok(rendered.includes('pending'));assert.ok(rendered.includes('status'));assert.equal(currentStateVersion,'current');assert.equal(buttons[0].disabled,false)}catch(e){console.error(e);process.exitCode=1}},25);
"""
    )
    path = tmp_path / "hung.js"
    path.write_text(program, encoding="utf-8")
    result = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
