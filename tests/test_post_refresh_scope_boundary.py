from unittest.mock import patch

from phase_agent.tools.workflows.scientific_module_scope import attach_module_scope


def test_first_post_refresh_generation_does_not_bundle_inputs():
    action = {"parameters": {"generation_cost_preview": {
        "selected_branch_upper": 20, "relax_mc_cost_upper": 100,
    }}}
    with patch("phase_agent.policies.model_generation.new_model_generation_facts",
               return_value={"required": True, "ready": False}):
        result = attach_module_scope(action, {}, {})
    assert "module_execution_scope" not in result["parameters"]


def test_registered_generation_keeps_normal_bundling():
    action = {"parameters": {"generation_cost_preview": {
        "selected_branch_upper": 20, "relax_mc_cost_upper": 100,
    }}}
    with patch("phase_agent.policies.model_generation.new_model_generation_facts",
               return_value={"required": True, "ready": True}):
        result = attach_module_scope(action, {}, {})
    assert result["parameters"]["module_execution_scope"]["submit_authorized"] is False
