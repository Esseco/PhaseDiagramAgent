from phase_agent.runtime.configuration_chat import _concise_import_reply


def test_import_reply_shows_only_actionable_missing_fields():
    reply = _concise_import_reply("O3=20", {"missing": ["remote_mlip"], "conflicts": []}, [])
    assert "remote_mlip" in reply and "读取配置 JSON" in reply
    assert "同意" not in reply


def test_passed_import_has_one_confirmation_instruction():
    reply = _concise_import_reply("O3=20", {}, [], agent_passed=True)
    assert reply.count("同意") == 1 and "修订" in reply
    assert len(reply.splitlines()) == 2
