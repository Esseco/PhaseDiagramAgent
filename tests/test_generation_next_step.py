from types import SimpleNamespace
from unittest.mock import patch
from execution_layer.state.restore_generation_gate import restore_generation_gate
from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
from decision_layer.agent.propose_tool_action import propose_agent_tool_action
from config_layer.defaults.default_budget_rules import default_budget_rules


def test_completed_generation_restores_gate_and_prepares_relax():
    manager = SimpleNamespace(data={"branches": {"B1": {"P": "O3", "structure_ids": ["S1"]}},
        "structures": {"S1": {"source_path": "mock.vasp", "composition": {"Na": 1, "O": 2},
            "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"}}}})
    state = {"generation_history": [{"registered_ids": ["S1"],
              "summary": {"unique_structures": 1, "registered_structures": 1}}]}
    recovered = restore_generation_gate(state, manager)
    assert "dedup_gate" not in state
    with patch("pathlib.Path.is_file", return_value=True):
        action = choose_debug_next_action(recovered, manager,
            {"upload_batches_directory": "upload", "budgets": default_budget_rules(),
             "mlip": {"name": "mh1", "model_path": "/model"}},
            allowed_tools=["generate_branches", "prepare_local_batch_files"], user_message="继续")
    assert action["tool"] == "prepare_local_batch_files"
    assert action["target_ids"] == ["B1"]


def test_pending_remote_dedup_is_not_overridden():
    state = {"dedup_gate": {"status": "pending"}}
    assert restore_generation_gate(state, SimpleNamespace(data={})) == state


def test_missing_evidence_does_not_declare_dedup_completed():
    state = {"generation_history": [{"registered_ids": ["S1"]}]}
    assert "dedup_gate" not in restore_generation_gate(state, SimpleNamespace(data={"structures": {"S1": {}}}))


def test_registered_nested_action_is_supported():
    result = propose_agent_tool_action({}, allowed_tools=["prepare_local_batch_files"],
        agent_client=lambda _: {"action": {"tool": "prepare_local_batch_files", "parameters": {}},
                                "_llm_usage": {"calls": 1}})
    assert result["tool"] == "prepare_local_batch_files"
    assert result["_llm_usage"]["calls"] == 1
