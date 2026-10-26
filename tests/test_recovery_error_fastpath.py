"""Read-only shortcuts surface saved-result recovery failures."""

import pytest

from tests.test_skill_recovery_boundaries import recovery_handler


@pytest.mark.parametrize("message", ["当前状态", "查看方案"])
def test_read_only_shortcut_keeps_recovery_error_visible(tmp_path, monkeypatch, message):
    handler, _requests, _state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "只读状态"}
    )
    reply = handler([{"role": "user", "content": message}], conversation_id="c")
    assert "结果回收失败" in reply
    assert "synthetic invalid return" in reply
