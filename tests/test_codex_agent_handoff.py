"""Codex's request-local callback must not replace the original Agent client."""

from pathlib import Path
from threading import RLock
from unittest.mock import Mock, patch

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl


@pytest.mark.parametrize("failure", [False, True])
def test_agent_continue_after_codex_proposal_uses_original_model(tmp_path, failure):
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text('{"execution_mode":"debug","deepseek":{"model":"original"}}', encoding="utf-8")
    original_config = runtime.read_bytes()
    model = Mock(return_value="original Agent continued")

    class Handler:
        def __init__(self):
            self.state_path = Path(tmp_path) / "state.json"
            self.workflow_kwargs = {"agent_client": model}
            self.lock = RLock()

        def _run(self, *args):
            callback = self.workflow_kwargs["agent_client"]
            assert callback is not model
            assert callback({})["tool"] == "pause_search"
            if failure:
                raise RuntimeError("synthetic Codex failure")
            return {"status": "paused", "submitted": False}

        def __call__(self, messages, *, conversation_id):
            return self.workflow_kwargs["agent_client"](messages)

    handler = Handler()
    control = LocalAgentControl(handler)
    snapshot = {"context_version": "V1", "context_digest": "D1", "allowed_tools": ["pause_search"], "decision_context": {}, "state": {}}
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=snapshot):
        if failure:
            with pytest.raises(RuntimeError, match="synthetic Codex"):
                control.propose_codex({"tool": "pause_search"}, context_version="V1", context_digest="D1")
        else:
            control.propose_codex({"tool": "pause_search"}, context_version="V1", context_digest="D1")
    assert handler.workflow_kwargs["agent_client"] is model
    model.assert_not_called()
    assert runtime.read_bytes() == original_config
    result = control.propose("继续", conversation_id="original-agent")
    assert result["reply"] == "original Agent continued"
    model.assert_called_once_with([{"role": "user", "content": "继续"}])
