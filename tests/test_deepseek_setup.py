import json
import http.client
import threading
import unittest
from unittest.mock import patch

from run.agent_api import create_server
from run.deepseek_setup import setup_page, test_and_save_api_key as _test_and_save_api_key


class DeepSeekSetupTest(unittest.TestCase):
    def test_setup_page_does_not_include_key_and_explains_local_storage(self):
        page = setup_page(model="deepseek-flash", configured=False)
        self.assertIn('type="password"', page)
        self.assertIn("deepseek-flash", page)
        self.assertIn("Windows 凭据管理器", page)
        self.assertNotIn("sk-test-secret", page)

    def test_valid_key_is_tested_then_saved_and_activated(self):
        activated = []
        client = lambda _payload: {"_llm_usage": {"calls": 1}}
        with patch("decision_layer.agent.create_deepseek_client.create_deepseek_client", return_value=client) as factory, \
                patch("run.deepseek_credentials.save_deepseek_api_key") as save:
            result = _test_and_save_api_key(
                "sk-test-secret-value",
                settings={"model": "deepseek-flash", "base_url": "https://api.deepseek.com"},
                activate_client=activated.append,
            )
        self.assertEqual(result, {"status": "connected", "model": "deepseek-flash"})
        self.assertEqual(factory.call_args.kwargs["max_tokens"], 32)
        save.assert_called_once_with("sk-test-secret-value")
        self.assertEqual(activated, ["sk-test-secret-value"])

    def test_failed_key_test_does_not_save_or_activate(self):
        activated = []
        with patch("decision_layer.agent.create_deepseek_client.create_deepseek_client", return_value=lambda _payload: {}), \
                patch("run.deepseek_credentials.save_deepseek_api_key") as save:
            with self.assertRaisesRegex(RuntimeError, "密钥未保存"):
                _test_and_save_api_key(
                    "sk-test-secret-value", settings={}, activate_client=activated.append,
                )
        save.assert_not_called()
        self.assertEqual(activated, [])

    def test_local_setup_http_page_and_post(self):
        configured = []
        server = create_server(
            lambda _messages, **_kwargs: "unused",
            api_key="local-test-token-long-enough",
            port=0,
            deepseek_model="deepseek-flash",
            deepseek_key_setup=lambda key: configured.append(key) or {
                "status": "connected", "model": "deepseek-flash",
            },
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            with patch("run.deepseek_credentials.load_deepseek_api_key", return_value=None):
                connection.request("GET", "/phase/setup")
                response = connection.getresponse()
                page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("启用相图 Agent", page)
            self.assertIn("deepseek-flash", page)
            payload = json.dumps({"api_key": "sk-test-secret-value"}).encode("utf-8")
            connection.request(
                "POST", "/phase/setup/key", body=payload,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertNotIn("sk-test-secret-value", json.dumps(result))
            self.assertEqual(result["status"], "connected")
            self.assertEqual(configured, ["sk-test-secret-value"])
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
