"""Isolation, CAS hits and whole-chain model limits without real providers."""

from copy import deepcopy
import json
import sqlite3
import threading
from unittest.mock import patch

import pytest

from phase_agent.runtime.api_deadline import (
    model_round,
    traced_model_call,
    consume_model_repair,
    current_model_round,
    ModelRepairLimitExceeded,
    ModelRoundDeadlineExceeded,
)
from phase_agent.runtime.turn_process import _current
from phase_agent.agents.common.post_dft_review import request_validated_action
from phase_agent.models.create_deepseek_client import create_deepseek_client
from phase_agent.agents.strategic.context import with_advice
from tests.test_on_demand_branch import context, review
from tests.test_streamed_checkpoint_blob import make


def repair_fixture():
    valid = review()
    valid.update(task_key="p", target_ids=[], budget=0, reason="review", expected_purpose="review")
    valid["parameters"].update(sampling_mode="region_adaptive", defer_reason="wait")
    bad = deepcopy(valid)
    bad["budget"] = -1
    refs = deepcopy(valid)
    refs["branch_pool_review"]["evidence_refs"] = ["invented"]
    return [bad, refs, valid]


def test_schema_branch_repairs_share_limit_and_preserve_known_usage():
    answers = repair_fixture()
    sent = []
    record = {}
    token = _current.set(record)

    def client(request):
        sent.append(request["mode"])
        return {
            **deepcopy(answers[len(sent) - 1]),
            "_llm_usage": {"calls": 1, "input_tokens": 2, "output_tokens": 1, "cost": None},
        }

    try:
        with model_round(repairs=1):
            with pytest.raises(ModelRepairLimitExceeded) as caught:
                request_validated_action(
                    client,
                    {
                        "mode": "autonomous_search",
                        "allowed_tools": ["generate_branches"],
                        "state": {},
                        "decision_context": context(),
                    },
                )
        assert sent == ["autonomous_search", "proposal_contract_repair"]
        assert caught.value.llm_usage == {
            "calls": 2,
            "input_tokens": 4,
            "output_tokens": 2,
            "cost": None,
        }
        chain = record["model_call_chain"]
        assert [row["reason"] for row in chain] == sent + ["branch_review_repair"]
        assert chain[1]["parent_call_id"] == chain[0]["call_id"]
        assert chain[2]["status"] == "ModelRepairLimitExceeded"
        assert all(row["seconds"] >= 0 for row in chain)
    finally:
        _current.reset(token)


def test_default_two_repairs_accepts_same_scientific_output_without_editing_it():
    answers = repair_fixture()
    sent = []

    def client(request):
        sent.append(request["mode"])
        return deepcopy(answers[len(sent) - 1])

    original = deepcopy(answers)
    result = request_validated_action(
        client,
        {
            "mode": "autonomous_search",
            "allowed_tools": ["generate_branches"],
            "state": {},
            "decision_context": context(),
        },
    )
    assert len(sent) == 3 and answers == original
    assert result["branch_pool_review"] == original[-1]["branch_pool_review"]
    assert current_model_round() is None


def test_chain_deadline_not_renewed_by_nested_round_and_failed_usage_retained():
    now = [100.0]

    def client(request):
        now[0] += 6
        return {"tool": "pause_search", "_llm_usage": {"calls": 1, "cost": None}}

    with patch("phase_agent.runtime.api_deadline.monotonic", side_effect=lambda: now[0]):
        with model_round(timeout=10, repairs=2) as outer:
            traced_model_call(client, {})
            with model_round(timeout=999, repairs=99) as inner:
                assert inner is outer and inner["deadline"] == 110
                with pytest.raises(ModelRoundDeadlineExceeded) as caught:
                    traced_model_call(client, {}, repair=True)
                assert caught.value.llm_usage["calls"] == 1
            assert len(outer["calls"]) == 2
    assert current_model_round() is None


class Response:
    def __init__(self, content, finish="stop"):
        self.content, self.finish = content, finish

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"finish_reason": self.finish, "message": {"content": self.content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ).encode()


def test_provider_truncation_consumes_same_repair_allowance_and_logs_physical_bytes():
    record = {"id": "fake-turn", "events": [], "timings": [], "model_requests": []}
    token = _current.set(record)
    client = create_deepseek_client(api_key="fake", thinking="enabled")
    try:
        with model_round(repairs=1):
            with patch(
                "urllib.request.urlopen",
                side_effect=[Response("", "length"), Response('{"tool":"pause_search"}')],
            ) as provider:
                result = client(
                    {
                        "mode": "autonomous_search",
                        "decision_kind": "strategy",
                        "allowed_tools": ["pause_search"],
                    }
                )
            assert provider.call_count == 2 and result["_llm_usage"]["calls"] == 2
            with pytest.raises(ModelRepairLimitExceeded):
                consume_model_repair("schema_repair")
        assert all(
            row["input_bytes"] > 0 and row["seconds"] >= 0 for row in record["model_requests"]
        )
        assert all(
            row["logical_call_id"] == record["model_call_chain"][0]["call_id"]
            for row in record["model_requests"]
        )
    finally:
        _current.reset(token)


def test_provider_deadlines_are_shared_across_calls():
    now = [100.0]
    deadlines = []

    def response(request, socket_timeout, deadline):
        deadlines.append(deadline)
        now[0] += 3
        return Response('{"tool":"pause_search"}').read()

    client = create_deepseek_client(api_key="fake", thinking="disabled", total_timeout=120)
    with patch("phase_agent.runtime.api_deadline.monotonic", side_effect=lambda: now[0]):
        with model_round(timeout=10):
            with patch("phase_agent.runtime.api_deadline.read_response", side_effect=response):
                client({"user_instruction": "first"})
                client({"user_instruction": "second"})
    assert deadlines == [110, 110]


def test_turn_shares_budget_between_request_chains_but_next_turn_does_not():
    for _ in range(2):
        record = {}
        token = _current.set(record)
        try:
            with model_round(repairs=1):
                consume_model_repair("schema")
            with model_round(repairs=99):
                with pytest.raises(ModelRepairLimitExceeded):
                    consume_model_repair("branch")
            assert record["_decision_model_round"]["repairs"] == 1
        finally:
            _current.reset(token)


def test_advice_context_copied_once_and_requests_remain_independent():
    copies = []

    class Counted(dict):
        def __deepcopy__(self, memo):
            copies.append(1)
            return deepcopy(dict(self), memo)

    payload = {
        "state": {"tasks": [{"id": "t"}]},
        "decision_context": Counted(rows=[{"id": "e"}]),
        "instruction": "goal",
    }
    before = deepcopy(dict(payload))
    copies.clear()
    advice = [{"snapshot_id": "s1", "assessment": "one"}]
    first = with_advice(payload, {"snapshot_id": "s1", "facts": {}}, advice)
    assert len(copies) == 1
    second = with_advice(payload, {"snapshot_id": "s2", "facts": {}}, [])
    first["decision_context"]["rows"][0]["id"] = "changed"
    first["state"]["tasks"].append({"id": "new"})
    first["decision_context"]["expert_advice"][0]["assessment"] = "changed"
    assert payload == before and advice[0]["assessment"] == "one"
    assert second["decision_context"]["rows"][0]["id"] == "e"
    assert second["decision_context"]["expert_advice"] == []


def test_advice_does_not_modify_other_aliases_of_original_context():
    context = {"rows": [{"id": "e"}]}
    p = {"decision_context": context, "alias": context}
    request = with_advice(p, {"snapshot_id": "s", "facts": {}}, [])
    assert "expert_advice" not in request["alias"]
    request["decision_context"]["rows"][0]["id"] = "changed"
    assert request["alias"]["rows"][0]["id"] == "e" and context["rows"][0]["id"] == "e"


def test_cas_hit_reencodes_and_rehashes_but_never_compresses_again():
    import phase_agent.persistence.checkpoint_blobs as blobs

    rows = [{"id": i, "payload": "x" * 512} for i in range(1400)]
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        first = serde._state({"rows": rows})
        db.commit()
        original_hash = blobs.sha256
        with (
            patch.object(
                blobs.zlib, "compressobj", side_effect=AssertionError("hit must not compress")
            ),
            patch.object(blobs, "sha256", wraps=original_hash) as digest,
        ):
            assert serde._state({"rows": rows}) == first
            assert digest.call_count == 1
        rows[0]["payload"] = "changed"
        changed = serde._state({"rows": rows})
        assert changed != first and serde._restore_state(first)["rows"][0]["payload"] != "changed"


def test_cas_rollback_invalidates_earlier_hit_and_reinserts_safely():
    rows = [{"id": i, "payload": "x" * 512} for i in range(1400)]
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        first = serde._state({"rows": rows})
        assert serde._state({"rows": rows}) == first
        db.rollback()
        recovered = serde._state({"rows": rows})
        db.commit()
        assert serde._restore_state(recovered)["rows"] == rows


def test_turn_diagnostics_do_not_persist_runtime_budget(tmp_path):
    from phase_agent.runtime.turn_process import record_turn

    cfg = tmp_path / "agent_runtime.json"
    cfg.write_text("{}")
    with record_turn(cfg, "fake request", "test") as record:
        traced_model_call(lambda p: {"answer": "fake"}, {"mode": "test"})
    assert "_decision_model_round" not in record
    loaded = json.loads(
        next((tmp_path / "workflow_state" / "dialogue_process").glob("*.json")).read_text(
            encoding="utf8"
        )
    )
    assert loaded["model_call_chain"][0]["usage"] is None


def test_provider_limit_failure_retains_usage_of_truncated_response():
    client = create_deepseek_client(api_key="fake", thinking="enabled")
    with model_round(repairs=0):
        with patch("urllib.request.urlopen", return_value=Response("", "length")) as provider:
            with pytest.raises(ModelRepairLimitExceeded) as caught:
                client({"decision_kind": "strategy", "allowed_tools": ["pause_search"]})
        assert provider.call_count == 1
        assert caught.value.llm_usage["calls"] == 1
        assert caught.value.llm_usage["input_tokens"] == 3
        assert caught.value.llm_usage["output_tokens"] == 2
        assert caught.value.llm_usage["cost"] is None


def test_client_adapters_preserve_model_round_policy():
    from phase_agent.models.capabilities import copy_model_capabilities

    client = create_deepseek_client(api_key="fake", total_timeout=17)
    wrapped = copy_model_capabilities(client, lambda payload: client(payload))
    assert wrapped.decision_timeout_seconds == 17
    assert wrapped.decision_repair_limit == 2


def test_one_correction_reports_schema_and_branch_errors_together():
    bad, refs, valid = repair_fixture()
    bad["branch_pool_review"]["evidence_refs"] = ["invented"]
    sent = []

    def client(request):
        sent.append(request)
        if len(sent) == 1:
            return deepcopy(bad)
        return deepcopy(
            valid
            if any("invented" in str(error) for error in request.get("validation_errors", []))
            else refs
        )

    result = request_validated_action(
        client,
        {
            "mode": "autonomous_search",
            "allowed_tools": ["generate_branches"],
            "state": {},
            "decision_context": context(),
        },
    )
    assert len(sent) == 2
    assert any("budget" in error for error in sent[1]["validation_errors"])
    assert any("invented" in error for error in sent[1]["validation_errors"])
    assert result["branch_pool_review"] == valid["branch_pool_review"]


def test_aggregate_repair_still_handles_non_object_output():
    from phase_agent.agents.common.proposal_schema_repair import repair_proposal_schema

    valid = repair_fixture()[-1]
    corrected = repair_proposal_schema(
        lambda request: deepcopy(valid), {"decision_context": context()}, []
    )
    assert corrected == valid
