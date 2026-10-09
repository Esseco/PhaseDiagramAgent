import pytest
from unittest.mock import patch
from run.agent_api import RunWorkflowChatHandler
from analysis_layer.cost.predict_runtime import predict_runtime
from analysis_layer.state.build_decision_context import build_decision_context


@pytest.mark.parametrize("steps", [None, 0, -1, True, float("nan")])
def test_mc_runtime_requires_positive_step_estimate(steps):
    result = predict_runtime("deep_search", atom_count=40, state={}, mc_steps=steps)
    assert result["status"] == "insufficient_parameters"
    assert result["reason"] == "positive_mc_steps_required"


def test_relax_stage_selects_mc_memory_scope():
    result = build_decision_context({"tasks": [{"stage": "relax_and_feature", "status": "completed"}]})
    assert result["memory_retrieval"]["action"] == "allocate_mc_bohb"


def test_model_switch_preserves_dedicated_intent_client(tmp_path):
    search, intent = object(), object()
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")},
        config_intent_client=intent,
        deepseek_model_switcher=lambda model: (model, search))
    with patch("run.resolve_deepseek_model_request.resolve_deepseek_model_request",
               return_value="deepseek-flash"):
        handler([{"role": "user", "content": "切换模型"}])
    assert handler.workflow_kwargs["agent_client"] is search
    assert handler.config_intent_client is intent
