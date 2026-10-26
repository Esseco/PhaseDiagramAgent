"""Approval page retries reuse one key until the concrete POST succeeds."""

import ast
import json
import subprocess
from pathlib import Path


def _approval_script():
    source_path = Path(__file__).parents[1] / "phase_agent" / "runtime" / "agent_api.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_approval_page"
    )
    page = ast.literal_eval(function.body[0].value)
    start = page.index("<script>") + len("<script>")
    return page[start : page.index("</script>", start)]


def test_approval_post_key_is_stable_until_success():
    script = _approval_script()
    node = f"""
const vm = require('vm');
const assert = require('assert');
const storage = new Map();
let uuids = 0;
const context = {{
  sessionStorage: {{
    getItem: key => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key),
  }},
  crypto: {{ randomUUID: () => 'uuid-' + (++uuids) }},
  document: {{ querySelectorAll: () => [] }},
}};
vm.runInNewContext({json.dumps(script)}, context);
const options = {{method: 'POST', body: JSON.stringify({{plan_id: 'p', decision: 'approve'}})}};
const first = context.idempotencyRecord('/phase/decision', options);
const retry = context.idempotencyRecord('/phase/decision', options);
assert.strictEqual(first.key, retry.key);
const other = context.idempotencyRecord('/phase/decision', {{method: 'POST', body: JSON.stringify({{plan_id: 'p', decision: 'reject'}})}});
assert.notStrictEqual(first.key, other.key);
first.clear();
const afterSuccess = context.idempotencyRecord('/phase/decision', options);
assert.notStrictEqual(first.key, afterSuccess.key);
"""
    result = subprocess.run(["node", "-e", node], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
