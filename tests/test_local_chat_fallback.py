import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import mock_open, patch

from run.local_project_launcher import _ensure_webui
from run.open_webui_api import create_server
from run.open_webui_startup import effective_webui_config


class LocalChatFallbackTests(unittest.TestCase):
    def test_local_chat_works_without_open_webui(self):
        seen = []

        def reply(messages, *, conversation_id):
            seen.append((messages[-1]["content"], conversation_id))
            return "已收到"

        server = create_server(reply, api_key="local-test-token-123456", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/phase/chat", timeout=3) as response:
                page = response.read().decode("utf-8")
            self.assertIn("相图搜索 Agent", page)
            self.assertIn("/v1/chat/completions", page)
            payload = json.dumps({"messages": [{"role": "user", "content": "查看状态"}],
                                  "metadata": {"chat_id": "fallback-test"}}).encode("utf-8")
            request = Request(base + "/v1/chat/completions", data=payload, headers={
                "Authorization": "Bearer local-test-token-123456",
                "Content-Type": "application/json",
            })
            with urlopen(request, timeout=3) as response:
                result = json.load(response)
            self.assertEqual(result["choices"][0]["message"]["content"], "已收到")
            self.assertEqual(seen, [("查看状态", "fallback-test")])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_shared_startup_settings_apply_to_replaced_projects(self):
        shared = {"open_webui_start_command": ["webui", "serve"],
                  "open_webui_workdir": "E:/shared-webui"}
        with patch("run.open_webui_startup.load_startup_settings", return_value=shared):
            first = effective_webui_config({"state_path": "old/state.json"})
            second = effective_webui_config({"state_path": "new/state.json"})
        self.assertEqual(first["open_webui_start_command"], second["open_webui_start_command"])
        self.assertEqual(second["open_webui_workdir"], "E:/shared-webui")

    def test_successful_one_shot_webui_start_waits_for_page(self):
        class FinishedCommand:
            def poll(self):
                return 0

        config = {"open_webui_start_command": ["webui", "start"],
                  "open_webui_startup_timeout_seconds": 5}
        with patch("run.local_project_launcher._webui_is_available", side_effect=[False, False, True]), \
             patch("run.local_project_launcher.subprocess.Popen", return_value=FinishedCommand()), \
             patch("run.local_project_launcher.time.sleep"), \
             patch("run.local_project_launcher.Path.is_dir", return_value=True), \
             patch("run.local_project_launcher.Path.open", mock_open()):
            ready, owned = _ensure_webui(config, Path("E:/example"), "http://127.0.0.1:3000")
        self.assertTrue(ready)
        self.assertIsNone(owned)


if __name__ == "__main__":
    unittest.main()
