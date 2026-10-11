from types import SimpleNamespace

import pytest

from phase_agent.graphs.dialogue.history import restore_session


@pytest.mark.parametrize("message", ["同意", "同意，直接生成", "读取配置 JSON", "查看当前设置"])
def test_bound_project_restores_without_consuming_message(message):
    handler = SimpleNamespace(history_decision=None, workflow_kwargs={}, new_run_factory=None)
    assert restore_session(handler, message, {"confirmed_config_version": "config-16"}) is None
    assert handler.history_decision == "continue"


def test_unbound_history_still_requires_selection():
    handler = SimpleNamespace(history_decision=None, workflow_kwargs={}, new_run_factory=None)
    assert "请明确回复" in restore_session(handler, "同意", {})
    assert handler.history_decision is None
