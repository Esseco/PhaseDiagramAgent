from run.agent_api import RunWorkflowChatHandler


def test_new_run_adopts_all_runtime_callbacks_without_replacing_lock(tmp_path):
    revision, switch, new_run, intent = (object() for _ in range(4))
    replacement = RunWorkflowChatHandler({"state_path": str(tmp_path / "new/state.json")},
        config_revision_factory=revision, deepseek_model_switcher=switch,
        new_run_factory=new_run, config_intent_client=intent)
    current = RunWorkflowChatHandler({"state_path": str(tmp_path / "old/state.json")},
        history_prompt=True, new_run_factory=lambda: replacement)
    original_lock = current.lock
    reply = current([{"role": "user", "content": "新建"}], conversation_id="chat")
    assert "已新建独立运行" in reply
    assert current.state_path == replacement.state_path
    assert current.config_revision_factory is revision
    assert current.deepseek_model_switcher is switch
    assert current.new_run_factory is new_run
    assert current.config_intent_client is intent
    assert current.lock is original_lock
    assert current.conversation_id == "chat"
