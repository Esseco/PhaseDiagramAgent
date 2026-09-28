import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run.local_project_launcher import (
    _ensure_webui, _stop_owned_process, create_project, describe_project,
    read_project_registry, register_project, start_agent,
)
from run.local_service_tokens import local_service_tokens
from run.open_webui_api import create_open_webui_runtime
from run.configuration_chat import ConfigurationChatHandler


class LocalProjectLauncherTests(unittest.TestCase):
    def test_existing_webui_is_not_owned_by_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("run.local_project_launcher._webui_is_available", return_value=True), \
                 patch("run.local_project_launcher.subprocess.Popen") as popen:
                self.assertEqual(_ensure_webui({}, Path(directory), "http://127.0.0.1:3000"),
                                 (True, None))
                popen.assert_not_called()

    def test_launcher_tracks_only_webui_process_it_started(self):
        class FakeProcess:
            def __init__(self):
                self.running = True

            def poll(self):
                return None if self.running else 0

        with tempfile.TemporaryDirectory() as directory:
            process = FakeProcess()
            config = {
                "open_webui_start_command": ["open-webui", "serve"],
                "open_webui_startup_timeout_seconds": 5,
            }
            with patch("run.local_project_launcher._webui_is_available",
                       side_effect=[False, True]), \
                 patch("run.local_project_launcher.subprocess.Popen", return_value=process):
                ready, owned = _ensure_webui(config, Path(directory), "http://127.0.0.1:3000")
            self.assertTrue(ready)
            self.assertIs(owned, process)

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
            session = json.loads((target.parent / "config_session.json").read_text(encoding="utf-8"))
            self.assertEqual(session["setup_stage"], "json_ready")
            self.assertEqual(session["status"], "draft")
            self.assertTrue((target.parent / "search_config.draft.json").is_file())
            self.assertIn("项目长期记忆：无", describe_project(target))
            with patch("run.deepseek_credentials.load_deepseek_api_key", return_value=None):
                handler = create_open_webui_runtime(target)
            self.assertIsInstance(handler, ConfigurationChatHandler)
            self.assertEqual(handler.workflow_kwargs["config_session"]["setup_stage"], "json_ready")
            self.assertEqual(handler.editable_config_path, target.parent / "search_config.draft.json")
            self.assertEqual(register_project(target, registry=registry), [str(target)])
            with self.assertRaises(FileExistsError):
                create_project(target.parent, registry=registry)

    def test_rejects_same_state_across_different_project_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            first = create_project(root / "a", registry=registry)
            second = root / "b" / "open_webui_runtime.json"
            second.parent.mkdir()
            second.write_text(json.dumps({"state_path": str(first.parent / "current/state.json"),
                                          "ledger_path": "other.json"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "同一 state"):
                register_project(second, registry=registry)

    def test_occupied_port_never_launches_other_project(self):
        with tempfile.TemporaryDirectory() as directory:
            target = create_project(Path(directory) / "a", registry=Path(directory) / "registry.json")
            with patch("run.local_project_launcher._port_is_free", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "占用"):
                    start_agent(target)

    def test_reuses_only_same_project_with_valid_connection_token(self):
        with tempfile.TemporaryDirectory() as directory:
            target = create_project(Path(directory) / "a", registry=Path(directory) / "registry.json")
            existing = object()
            with patch("run.local_project_launcher._port_is_free", return_value=False), \
                 patch("run.local_project_launcher._existing_agent", return_value=existing), \
                 patch("run.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)), \
                 patch("run.local_project_launcher._agent_accepts_token", return_value=True):
                self.assertEqual(start_agent(target), (existing, "a" * 24, True))
            with patch("run.local_project_launcher._port_is_free", return_value=False), \
                 patch("run.local_project_launcher._existing_agent", return_value=existing), \
                 patch("run.local_project_launcher.local_service_tokens", return_value=("a" * 24, "b" * 24)), \
                 patch("run.local_project_launcher._agent_accepts_token", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "不匹配"):
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
