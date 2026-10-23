import json

from phase_agent.persistence.memory.conversation import memory_path, recent_turns, remember_turn
from tests.test_unified_react_dialogue import make_handler


def test_malformed_memory_row_does_not_break_valid_history_or_answer(tmp_path):
    path = tmp_path / "state.json"
    memory = memory_path(path, "t")
    memory.parent.mkdir()
    memory.write_text(json.dumps([None, {"user": "why", "assistant": "known failure"}, 2]))
    assert recent_turns(path, "t") == [{"user": "why", "assistant": "known failure"}]
    remember_turn(path, "t", "what next", "repair file")
    assert recent_turns(path, "t")[-1]["assistant"] == "repair file"
    assert recent_turns(path, "other") == []


def test_followup_context_uses_previous_reply_once_and_needs_one_model_call(tmp_path):
    calls = []
    handler = make_handler(
        tmp_path, lambda payload: calls.append(payload) or {"kind": "answer", "answer": "解释原因"}
    )
    previous = "Unique prior answer identifies the file to repair."
    remember_turn(handler.state_path, "t", "what happened", previous)
    answer = handler([{"role": "user", "content": "具体怎么处理"}], conversation_id="t")
    assert answer == "解释原因" and len(calls) == 1
    facts = calls[0]["conversation_facts"]
    assert facts["previous_reply"] == previous
    assert json.dumps(facts).count(previous) == 1


def test_supplied_conversation_survives_missing_memory_file(tmp_path):
    calls = []
    handler = make_handler(
        tmp_path, lambda payload: calls.append(payload) or {"kind": "answer", "answer": "可以解释"}
    )
    handler(
        [
            {"role": "user", "content": "what happened"},
            {"role": "assistant", "content": "Specific previous file problem"},
            {"role": "user", "content": "为什么"},
        ],
        conversation_id="fresh",
    )
    assert calls[0]["conversation_facts"]["previous_reply"] == "Specific previous file problem"


def test_context_read_does_not_load_same_config_session_twice(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.support import fresh_turn_context
    from phase_agent.runtime import project_context

    handler = make_handler(tmp_path, lambda payload: {"kind": "answer", "answer": "ok"})
    original = project_context.current_config_session
    calls = []

    def read(handler):
        calls.append(1)
        return original(handler)

    monkeypatch.setattr(project_context, "current_config_session", read)
    fresh_turn_context(handler, {}, [], "t")
    assert len(calls) == 1


def test_reviewed_memory_keeps_scope_and_model_rules_with_invalid_neighbors():
    from phase_agent.persistence.memory.retrieve_relevant_knowledge import (
        retrieve_relevant_knowledge,
    )

    common = {
        "scope": "user",
        "status": "active",
        "approved_by_user": True,
        "maturity": "heuristic",
        "category": "search",
    }
    records = [
        None,
        1,
        {**common, "knowledge_id": "old", "statement": "old guidance"},
        {**common, "knowledge_id": "new", "statement": "new guidance", "supersedes": ["old"]},
        {
            **common,
            "knowledge_id": "wrong-model",
            "statement": "old model",
            "model_dependent": True,
            "model_version": "old",
        },
        {
            **common,
            "knowledge_id": "private",
            "statement": "other project",
            "scope": "project",
            "system_id": "other",
        },
    ]
    state = {"decision_memory": {"records": records}}
    assert [
        row["knowledge_id"]
        for row in retrieve_relevant_knowledge(state, system_id="current", model_version="new")
    ] == ["new"]


def test_conversation_storage_does_not_depend_on_graph_or_runtime():
    import ast
    from pathlib import Path
    from phase_agent.persistence.memory import conversation
    import phase_agent.text_safety as text_safety

    for module in (conversation, text_safety):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        dependencies = [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        dependencies.extend(
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        )
        assert not any(
            name.startswith(("phase_agent.graphs", "phase_agent.runtime")) for name in dependencies
        )


def test_persisted_history_redacts_credentials_and_keeps_thread_boundary(tmp_path):
    path = tmp_path / "state.json"
    remember_turn(path, "a", "api_key=sk-private-credential", "Bearer private-token")
    raw = memory_path(path, "a").read_text(encoding="utf-8")
    assert "sk-private-credential" not in raw and "private-token" not in raw
    assert recent_turns(path, "a") and recent_turns(path, "b") == []


def test_studio_memory_cache_eviction_keeps_persistent_conversation(tmp_path, monkeypatch):
    from phase_agent.runtime import studio_session_scope

    monkeypatch.setattr(studio_session_scope, "is_project_session", lambda: True)
    monkeypatch.setattr(studio_session_scope, "thread_history", lambda: None)
    handler = make_handler(tmp_path, lambda payload: None)
    supplied = []

    def respond(messages, **kwargs):
        supplied.append(messages)
        handler.last_turn_outcome = "answered"
        return "Earlier reply " + kwargs["conversation_id"]

    handler._respond = respond
    for index in range(34):
        handler([{"role": "user", "content": "Explain saved facts"}], conversation_id=str(index))
    assert len(handler.studio_dialogues) == 32 and "0" not in handler.studio_dialogues
    handler([{"role": "user", "content": "Then what?"}], conversation_id="0")
    assert any(row.get("content") == "Earlier reply 0" for row in supplied[-1])
    assert len(handler.studio_dialogues) == 32
