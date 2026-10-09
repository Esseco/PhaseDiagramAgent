import json
import pytest
from run.artifact_queries import finetune_location_reply
from run.chat_application import RunWorkflowChatHandler


@pytest.mark.parametrize("message", ["微调文件夹在哪", "训练输入在哪里？", "看一下微调路径", "committee目录呢"])
def test_location_does_not_dispatch_or_modify_pending(tmp_path, message):
    directory = tmp_path / "MLIP-finetune-round-0001"
    directory.mkdir()
    state = {"active_model_version": "m1", "remote_finetune_jobs": {"j": {
        "directory": str(directory), "original_model_version": "m1"}},
        "pending_execution_policies": {"p": {}},
        "finetune_input_conflict": {"directory": str(directory)}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Location query must not call workflow or LLM")
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": forbidden},
                                     workflow=forbidden, history_prompt=True)
    reply = handler([{"role": "user", "content": message}])
    assert str(directory) in reply
    assert "旧输入待确认" in reply
    assert path.read_bytes() == before
    assert "本轮总结" not in reply


def test_absent_directory_and_no_record(tmp_path):
    assert "尚无" in finetune_location_reply("微调文件夹在哪", {})
    state = {"remote_finetune_jobs": {"j": {"directory": str(tmp_path / "missing")}}}
    assert "目录不存在" in finetune_location_reply("微调文件夹在哪", state)


@pytest.mark.parametrize("message", ["重新生成微调文件夹", "提交这个训练目录", "删除微调目录", "同意微调"])
def test_actions_are_not_location_queries(message):
    assert finetune_location_reply(message, {}) is None
