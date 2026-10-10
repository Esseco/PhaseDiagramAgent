import tempfile
import unittest

from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.session.load_config_session import load_config_session
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.configuration.session.save_config_session import save_config_session


class AutonomousPlatformTest(unittest.TestCase):
    def configured(self):
        session = create_config_draft(default_layered_search_config())
        session = apply_config_revision(session, {"calculation.mlip_version": "m1", "dft.parameters": {"encut": 520}, "convergence.hull_tolerance": .01, "convergence.minimum_dft_validations": 2, "convergence.coverage_threshold": .9})
        return session

    def test_unconfirmed_is_blocked_and_session_resumes(self):
        session = self.configured(); registry = create_tool_registry()
        result = run_tool_step(None, session, registry=registry, execute=True, invocation_id="x")
        self.assertEqual(result["status"], "rejected")
        self.assertIn("configuration_not_confirmed", result["validation"]["errors"])
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = f"{directory}/draft.json"; save_config_session(session, path); loaded = load_config_session(path)
        self.assertEqual(loaded["draft_revision"], session["draft_revision"])

    def test_invalid_model_action_is_blocked_without_rule_execution(self):
        session = confirm_config_snapshot(self.configured(), user_confirmed=True)
        calls = []
        registry = create_tool_registry({"check_convergence": lambda action, context: calls.append(action) or {"converged": False, "budget_exhausted": True}})
        invalid = lambda _: {"tool": "run_calculation_stage", "task_key": "T1", "parameters": {"parameters": {"encut": 300}, "dft.parameters": {}}, "budget": 1, "reason": "bad"}
        first = run_tool_step(None, session, registry=registry, agent_client=invalid, execute=True, invocation_id="a")
        self.assertEqual(first["status"], "not_configured")
        self.assertNotIn("execution", first)
        self.assertFalse(first["state"].get("pending_execution_policies"))
        repeated = run_tool_step(first["state"], session, registry=registry, agent_client=invalid, execute=True, invocation_id="a")
        self.assertEqual(repeated["status"], "not_configured")
        self.assertEqual(calls, [])
        self.assertEqual(repeated["state"]["confirmed_config_version"] if "confirmed_config_version" in repeated["state"] else session["confirmed_snapshot"]["config_version"], session["confirmed_snapshot"]["config_version"])


if __name__ == "__main__":
    unittest.main()
