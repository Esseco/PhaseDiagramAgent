import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
from phase_agent.decisions.agent.transport_tables import (
    compact_evidence_tables,
    expand_evidence_tables,
)
from phase_agent.graphs.dialogue.nodes import route_request
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.budget.record_budget_usage import record_budget_usage
from tests.test_hitl_closure import activation_state


def test_tables_are_lossless_and_keep_every_candidate_and_coverage_row():
    rows = [
        {
            "candidate_id": str(i),
            "predicted_Ehull": i / 100,
            "phase": "O3",
            "qbc": {"status": "unavailable"},
            "model_version": "m1",
        }
        for i in range(240)
    ]
    value = {
        "qbc_candidates": rows,
        "search_coverage": [{"P": "O3", "structure_count": i} for i in range(80)],
    }
    original = deepcopy(value)
    encoded = compact_evidence_tables(value)
    assert expand_evidence_tables(encoded) == original
    assert value == original
    assert len(json.dumps(encoded)) < len(json.dumps(value)) * 0.5
    assert encoded["qbc_candidates"]["count"] == 240


def test_missing_fields_remain_distinct_from_null():
    rows = [{"candidate_id": str(i), "metric": None} for i in range(8)] + [
        {"candidate_id": "missing"}
    ]
    decoded = expand_evidence_tables(compact_evidence_tables({"qbc_candidates": rows}))
    assert decoded["qbc_candidates"] == rows
    assert "metric" not in decoded["qbc_candidates"][-1]
    assert decoded["qbc_candidates"][0]["metric"] is None


def test_nested_training_evidence_compacts_without_touching_human_constraints():
    evidence = {"qbc_candidates": [{"candidate_id": str(i), "phase": "P3"} for i in range(20)]}
    payload = {
        "state": {},
        "decision_context": {
            "search_evidence": evidence,
            "long_term_memory": {"human_system_knowledge": ["固定Fe/Mn，不改变预算"]},
        },
    }
    decoded = expand_evidence_tables(prepare_llm_request(payload))
    assert decoded["decision_context"]["search_evidence"] == evidence
    assert (
        decoded["decision_context"]["long_term_memory"]
        == payload["decision_context"]["long_term_memory"]
    )


def test_dft_output_budget_counts_table_rows_not_table_keys():
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    from tests.test_deepseek_client import FakeResponse, completion

    client = create_deepseek_client(api_key="mock", thinking="disabled")
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        return_value=FakeResponse(completion('{"tool":"select_dft_candidates"}')),
    ) as request:
        client(
            {
                "allowed_tools": ["select_dft_candidates"],
                "state": {},
                "decision_context": {
                    "qbc_candidates": [{"candidate_id": str(i), "phase": "O3"} for i in range(100)]
                },
            }
        )
    body = json.loads(request.call_args.args[0].data)
    assert body["max_tokens"] >= 11024


def test_pending_activation_continue_shows_saved_plan_without_scientific_call(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PHASE_CONTROL_PORT", "8767")
    facts = activation_state()
    handler = SimpleNamespace(
        conversation_id="t", config_delegate=None, state_path=tmp_path / "state.json"
    )
    before = deepcopy(facts)
    runtime = SimpleNamespace(context=SimpleNamespace(handler=handler, conversation_id="t"))
    with patch(
        "phase_agent.decisions.agent.propose_tool_action.propose_agent_tool_action",
        Mock(side_effect=AssertionError("no LLM")),
    ):
        result = route_request({"facts": facts, "message": "继续"}, runtime)
    assert "已保存方案" in result["reply"]
    assert "8767/phase/approval" in result["reply"]
    assert facts == before
    assert handler.presented_plan_binding
    assert route_request({"facts": facts, "message": "同意"}, runtime)["operation"] == "review"
    assert route_request({"facts": facts, "message": "改为补DFT"}, runtime)["operation"] == "decide"


def test_compact_state_write_preserves_data_and_approval_identity(tmp_path):
    from phase_agent.tools.step_runner.build_status_summary import build_status_summary

    state = activation_state()
    state["structure"] = {"sites": [[i, 0.1, 0.2] for i in range(100)]}
    path = tmp_path / "state.json"
    write_json(path, state)
    assert read_json(path) == state
    assert build_status_summary(read_json(path)) == build_status_summary(state)
    assert path.stat().st_size < len(json.dumps(state, ensure_ascii=False, indent=2).encode()) * 0.7
    first_mtime = path.stat().st_mtime_ns
    write_json(path, state)
    assert path.stat().st_mtime_ns == first_mtime
    readable = tmp_path / "proposal.json"
    write_json(readable, state)
    assert b'\n  "' in readable.read_bytes()


def test_usage_breakdown_is_bounded_and_does_not_double_count():
    state = {}
    actual = {
        "llm_usage": {
            "calls": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "mode": "unified_dialogue",
        }
    }
    for _ in range(25):
        state = record_budget_usage(state, actual)
    llm = state["budget_usage"]["llm"]
    assert llm["calls"] == 25
    assert llm["input_tokens"] == 250
    assert llm["by_mode"]["unified_dialogue"]["input_tokens"] == 250
    assert len(llm["recent_calls"]) == 20
