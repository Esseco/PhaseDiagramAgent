from run.web_chat_gate import ChatRequestGate


def test_replay_and_busy_are_separate():
    gate = ChatRequestGate()
    assert gate.begin("chat", "one") is None
    assert "already" in gate.begin("chat", "one")
    assert "busy" in gate.begin("chat", "two")
    gate.finish()
    assert "already" in gate.begin("chat", "one")
    assert gate.begin("chat", "two") is None


def test_full_history_does_not_evict_replay_protection():
    gate = ChatRequestGate(capacity=1)
    assert "required" in gate.begin(None, "one")
    assert gate.begin("chat", "one") is None
    gate.finish()
    assert "full" in gate.begin("chat", "two")
    assert "already" in gate.begin("chat", "one")
