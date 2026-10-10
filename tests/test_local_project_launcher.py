import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase_agent.runtime.local_project_launcher import (
    _stop_owned_process, create_project, describe_project,
    read_project_registry, register_project, start_agent,
)
from phase_agent.runtime.local_service_tokens import local_service_tokens
from phase_agent.runtime.agent_api import create_agent_runtime
from phase_agent.runtime.configuration_chat import ConfigurationChatHandler


class LocalProjectLauncherTests(unittest.TestCase):
    def test_stop_terminates_only_passed_owned_process(self):
        class FakeProcess:
            def __init__(self, running):
                self.running = running
                self.terminated = False

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.terminated = True
                self.running = False

            def wait(self, timeout):
                return 0

            def kill(self):
                self.running = False

        owned = FakeProcess(running=True)
        self.assertTrue(_stop_owned_process(owned))
        self.assertTrue(owned.terminated)
        self.assertTrue(_stop_owned_process(None))

    def test_new_project_has_isolated_draft_and_memory_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            registry = base / "registry.json"
            target = create_project(base / "project_a", registry=registry)
            self.assertEqual(read_project_registry(registry), [str(target)])
            session = json.loads((target.parent / "parameters/config_session.json").read_text(encoding="utf-8"))
            self.assertEqual(session["setup_stage"], "json_ready")
            self.assertEqual(session["status"], "draft")
            self.assertTrue((target.parent / "parameters/search_config.project.json").is_file())
            self.assertIn("项目长期记忆：无", describe_project(target))
            with patch("phase_agent.runtime.deepseek_credentials.load_deepseek_api_key", return_value=None):
                handler = create_agent_runtime(target)
            self.assertIsInstance(handler, ConfigurationChatHandler)
            self.assertEqual(handler.workflow_kwargs["config_session"]["setup_stage"], "json_ready")
            self.assertEqual(handler.editable_config_path, target.parent / "parameters/search_config.project.json")
            self.assertEqual(register_project(target, registry=registry), [str(target)])
            with self.assertRaises(FileExistsError):
                create_project(target.parent, registry=registry)

    def test_rejects_same_state_across_different_project_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            first = create_project(root / "a", registry=registry)
            second = root / "b" / "agent_runtime.json"
            second.parent.mkdir()
            second.write_text(json.dumps({"state_path": str(first.parent / "workflow_state/state.json"),
                                          "ledger_path": "other.json"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "同一 state"):
                register_project(second, registry=registry)

    def test_occupied_port_never_launches_other_project(self):
        with tempfile.TemporaryDirectory() as directory:
            target = create_project(Path(directory) / "a", registry=Path(directory) / "registry.json")
            with patch("phase_agent.runtime.local_project_launcher._port_is_free", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "占用"):
                    start_agent(target)

    def test_never_reattaches_background_agent_even_with_valid_token(self):
        with tempfile.TemporaryDirectory() as directory:
            target = create_project(Path(directory) / "a", registry=Path(directory) / "registry.json")
            existing = object()
            with patch("phase_agent.runtime.local_project_launcher._port_is_free", return_value=False),\
                 patch("phase_agent.runtime.local_project_launcher._existing_agent", return_value=existing),\
                 patch("phase_agent.runtime.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)),\
                 patch("phase_agent.runtime.local_project_launcher._agent_accepts_token", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "占用"):
                    start_agent(target)
            with patch("phase_agent.runtime.local_project_launcher._port_is_free", return_value=False),\
                 patch("phase_agent.runtime.local_project_launcher._existing_agent", return_value=existing),\
                 patch("phase_agent.runtime.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)),\
                 patch("phase_agent.runtime.local_project_launcher._agent_accepts_token", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "占用"):
                    start_agent(target)

    def test_distinct_local_tokens_are_required(self):
        with patch.dict("os.environ", {"OPENWEBUI_TOOL_TOKEN": "a" * 24,
                                       "OPENWEBUI_CONTROL_TOKEN": "b" * 24}):
            self.assertEqual(local_service_tokens(), ("a" * 24, "b" * 24))
        with patch.dict("os.environ", {"OPENWEBUI_TOOL_TOKEN": "a" * 24,
                                       "OPENWEBUI_CONTROL_TOKEN": "a" * 24}):
            with self.assertRaisesRegex(RuntimeError, "必须不同"):
                local_service_tokens()


if __name__ == "__main__":
    unittest.main()
