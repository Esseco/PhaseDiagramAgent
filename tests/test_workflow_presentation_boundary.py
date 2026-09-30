import ast
from pathlib import Path

from run import workflow_reply_presentation as presentation
from run.chat_approval_rules import is_sensitive_proposal
from run.open_webui_api import format_workflow_reply


def test_old_public_reply_import_is_the_same_function():
    assert format_workflow_reply is presentation.format_workflow_reply
    assert format_workflow_reply({"status": "completed"}, "state.json") == "已完成：`workflow`。"


def test_presentation_does_not_import_runtime_or_http_entry():
    tree = ast.parse(Path(presentation.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "run.open_webui_api" not in imports
    assert "http.server" not in imports


def test_sensitive_classification_has_one_shared_source():
    assert is_sensitive_proposal({"raw_action": {"tool": "select_dft_candidates",
        "parameters": {"decisions": [{"action": "DFT_RELAX"}]}}})
    assert not is_sensitive_proposal({"raw_action": {"tool": "select_dft_candidates",
        "parameters": {"decisions": [{"action": "DFT_SINGLE_POINT"}]}}})
