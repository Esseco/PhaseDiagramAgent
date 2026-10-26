"""Final user receipts belong to their original turn, never scientific state."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.graphs.dialogue.support import fresh_turn_context
from phase_agent.persistence.memory.conversation import memory_path, recent_turns, remember_turn
from phase_agent.runtime.interaction_reports import save_interaction
from tests.test_unified_react_dialogue import make_handler


def test_identified_turn_upsert_updates_its_own_row_and_is_idempotent(tmp_path):
    state_path = tmp_path / "state.json"
    remember_turn(state_path, "a", "first", "initial", turn_id="first", facts_signature="saved")
    remember_turn(state_path, "a", "second", "later", turn_id="second")
    remember_turn(state_path, "b", "other", "private", turn_id="first")
    remember_turn(state_path, "a", "first", "final receipt /first.md", turn_id="first")
    before = memory_path(state_path, "a").read_bytes()
    remember_turn(state_path, "a", "first", "final receipt /first.md", turn_id="first")
    assert memory_path(state_path, "a").read_bytes() == before
    rows = recent_turns(state_path, "a")
    assert [row["turn_id"] for row in rows] == ["first", "second"]
    assert rows[0]["assistant"] == "final receipt /first.md"
    assert rows[0]["facts_signature"] == "saved"
    assert rows[1]["assistant"] == "later"
    assert recent_turns(state_path, "b")[0]["assistant"] == "private"


@pytest.mark.parametrize("display_matches_full", [True, False])
def test_markdown_has_short_conclusion_and_one_complete_body(tmp_path, display_matches_full):
    paragraph = "唯一的完整提交和回传证据。" * 200
    full = "结果：输入准备完毕。\n下一步：确认提交目录。\n" + paragraph
    display = full if display_matches_full else full + "\n预算：仍使用原批准预算。"
    record = {"id": "b" * 32, "thread_id": "a"}
    reply = save_interaction(tmp_path, "continue", full, display, {}, record=record)
    path = Path(record["interaction_report"])
    body = path.read_text(encoding="utf-8")
    conclusion = body.split("## 本轮结论\n\n", 1)[1].split("## 详细说明", 1)[0]
    assert len(conclusion) < 600 and paragraph not in conclusion
    assert body.count(paragraph) == 1
    assert body.count("## 详细说明") == 1
    second = save_interaction(tmp_path, "continue", full, display, {}, record=record)
    assert second == reply
    assert list((tmp_path / "interaction_records").rglob("*.md")) == [path]


@pytest.mark.parametrize("entry", ["studio", "control"])
@pytest.mark.parametrize("phase", ["submission", "return"])
def test_real_entry_remembers_final_receipt_in_same_turn_without_scientific_writes(
    tmp_path, monkeypatch, entry, phase
):
    import phase_agent.runtime.studio_runtime as runtime
    from phase_agent.runtime.agent_web_service import create_control_server

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    monkeypatch.setattr("phase_agent.runtime.loaded_code.assert_loaded_code_current", lambda: None)
    monkeypatch.setattr(
        "phase_agent.runtime.local_service_tokens.local_service_tokens",
        lambda: ("a" * 24, "b" * 24),
    )
    monkeypatch.setattr(
        "phase_agent.runtime.agent_api.create_server",
        lambda chat, **kwargs: SimpleNamespace(chat=chat),
    )
    handler = make_handler(tmp_path, lambda payload: pytest.fail("no model call authorized"))
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "tasks": [{"task_id": "task-1", "status": "pending", "relax_mc_bundle_id": "b1"}],
        "pending_execution_policies": {
            "original-plan": {"status": "awaiting_approval", "proposal_hash": "original-hash"}
        },
        "budget_reservations": {"original-plan": {"gpu_hours": 2}},
        "decision_memory": {"records": []},
    }
    if phase == "submission":
        from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

        directory = tmp_path / "submissions/epoch0/relax_mc/RelaxMC_Batch01"
        folder = directory.with_name(directory.name + "_input")
        folder.mkdir(parents=True)
        for name in ("submit_all.sh", "inputs.zip", "run_upload.py"):
            (folder / name).write_text("synthetic")
        state["relax_mc_batches"] = {"b1": {"directory": str(directory), "model_version": "m1"}}
        full = "\n".join(waiting_file_lines(state))
        assert "sbatch submit_all.sh" in full
    else:
        full = "结果：回传已核对；原方案仍待同意。\n下一步：查看记录后决定是否同意。"
    paragraph = "本轮唯一完整证据 " + phase + "。" * 500
    full += "\n" + paragraph
    handler.state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    before = handler.state_path.read_bytes()
    original = deepcopy(state)
    calls = []

    def respond(messages, *, conversation_id):
        calls.append((conversation_id, messages))
        handler.conversation_id = conversation_id
        handler.last_turn_outcome = "answered"
        return full

    monkeypatch.setattr(handler, "_respond", respond)
    monkeypatch.setattr(handler, "_run", lambda *args, **kwargs: pytest.fail("no execution"))
    if entry == "studio":
        monkeypatch.setattr(runtime, "_handler", handler)
        send = lambda text, thread: runtime.send_project_message(text, thread)
    else:
        server = create_control_server(handler, config, port=0)
        send = lambda text, thread: server.chat(
            [{"role": "user", "content": text}], conversation_id=thread
        )

    reply = send("核对本轮文件", "a")
    rows = recent_turns(handler.state_path, "a")
    assert len(rows) == 1 and rows[0]["assistant"] == reply
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "workflow_state/dialogue_process").glob("*.json")
    ]
    assert len(records) == 1 and rows[0]["turn_id"] == records[0]["id"]
    report = Path(records[0]["interaction_report"])
    assert str(report) in reply and report.is_file()
    assert report.read_text(encoding="utf-8").count(paragraph) == 1
    context = fresh_turn_context(handler, state, [], "a")
    assert context["previous_reply"] == reply
    assert handler.recent_dialogue[-1]["content"] == reply
    stored = memory_path(handler.state_path, "a").read_bytes()
    handler.remember_turn_reply(records[0], reply)
    assert memory_path(handler.state_path, "a").read_bytes() == stored
    assert len(recent_turns(handler.state_path, "a")) == 1
    assert recent_turns(handler.state_path, "other") == []
    assert handler.state_path.read_bytes() == before and state == original
    assert len(calls) == 1
    if entry == "studio":
        other = send("另一个会话的问题", "other")
        assert len(recent_turns(handler.state_path, "a")) == 1
        assert recent_turns(handler.state_path, "a")[0]["assistant"] == reply
        assert recent_turns(handler.state_path, "other")[0]["assistant"] == other
        assert str(report) not in other
        assert handler.studio_dialogues["a"][-1]["content"] == reply
        assert handler.state_path.read_bytes() == before


def test_studio_metadata_receipt_without_bound_memory_stays_out_of_history(tmp_path, monkeypatch):
    import phase_agent.runtime.studio_runtime as runtime
    from phase_agent.graphs.dialogue.commands import _open_webui_metadata_reply

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    monkeypatch.setattr("phase_agent.runtime.loaded_code.assert_loaded_code_current", lambda: None)
    handler = make_handler(tmp_path, lambda payload: pytest.fail("no model call"))
    handler.state_path.write_text("{}")
    before = handler.state_path.read_bytes()
    message = "### Task: Generate a concise title for this conversation"
    metadata = _open_webui_metadata_reply(message)
    assert metadata is not None
    monkeypatch.setattr(handler, "_respond", lambda messages, **kwargs: metadata)
    monkeypatch.setattr(runtime, "_handler", handler)
    runtime.send_project_message(message, "metadata-thread")
    assert recent_turns(handler.state_path, "metadata-thread") == []
    assert not memory_path(handler.state_path, "metadata-thread").exists()
    assert handler.recent_dialogue == [] and handler.studio_dialogues == {}
    assert handler.state_path.read_bytes() == before


def test_evicted_turn_receipt_cannot_append_or_replace_newer_cached_history(tmp_path):
    handler = make_handler(tmp_path, lambda payload: pytest.fail("no model call"))
    handler.state_path.write_text("{}")
    state_before = handler.state_path.read_bytes()
    for index in range(7):
        remember_turn(
            handler.state_path,
            "a",
            f"question {index}",
            f"answer {index}",
            turn_id=f"turn-{index}",
            facts_signature=f"facts-{index}",
        )
    assert [row["turn_id"] for row in recent_turns(handler.state_path, "a")] == [
        f"turn-{index}" for index in range(1, 7)
    ]
    handler.conversation_id = "a"
    handler.recent_dialogue = [
        {"role": "user", "content": "question 6"},
        {"role": "assistant", "content": "answer 6"},
    ]
    handler.studio_dialogues["a"] = deepcopy(handler.recent_dialogue)
    memory_before = memory_path(handler.state_path, "a").read_bytes()
    cache_before = deepcopy((handler.recent_dialogue, handler.studio_dialogues))
    record = {"id": "turn-0", "thread_id": "a", "message": "question 0"}
    handler.remember_turn_reply(record, "late receipt /old.md")
    assert memory_path(handler.state_path, "a").read_bytes() == memory_before
    assert (handler.recent_dialogue, handler.studio_dialogues) == cache_before
    assert handler.state_path.read_bytes() == state_before
