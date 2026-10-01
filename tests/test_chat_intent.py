from decision_layer.agent.resolve_chat_intent import resolve_chat_intent


def client(intent, **kwargs):
    return lambda payload: {"intent": intent, "direct_request": True, "confidence": 0.98, **kwargs}


def test_paraphrase_export():
    assert resolve_chat_intent("把最新凸包数据表给一下", {},
        agent_client=client("export_phase_csv"))["intent"] == "export_phase_csv"


def test_no_approval_authority():
    assert resolve_chat_intent("行吧", {}, agent_client=client("approve"))["intent"] == "unavailable"


def test_question_not_request():
    assert resolve_chat_intent("能导出吗", {}, agent_client=client("export_phase_csv",
        direct_request=False))["intent"] == "other"


def test_redo_requires_stage_and_round():
    assert resolve_chat_intent("那个再做一遍", {}, agent_client=client("redo_plan"))["intent"] == "clarify"
    assert resolve_chat_intent("本轮MC输入从头准备", {}, agent_client=client("redo_plan",
        stage="mc", scope="current"))["intent"] == "redo_plan"


def test_client_failure_is_explicit_and_does_not_guess_intent():
    def fail(payload):
        raise RuntimeError("offline")
    assert resolve_chat_intent("进度呢", {}, agent_client=fail)["intent"] == "unavailable"
