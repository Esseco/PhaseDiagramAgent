import json
import unittest
from unittest.mock import patch

from decision_layer.agent.create_deepseek_client import (
    DeepSeekResponseError,
    create_deepseek_client,
)


class FakeResponse:
    def __init__(self, value):
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def completion(content, *, finish_reason="stop", prompt_tokens=10, completion_tokens=4):
    return {
        "model": "deepseek-v4-pro",
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class DeepSeekClientTest(unittest.TestCase):
    def test_empty_json_output_is_retried_and_usage_counts_both_calls(self):
        client = create_deepseek_client(api_key="test-secret-do-not-print", thinking="disabled")
        with patch(
            "decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
            side_effect=[FakeResponse(completion("")), FakeResponse(completion('{"reply":"ok"}'))],
        ) as urlopen:
            result = client({"instruction": "test"})

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(result["reply"], "ok")
        self.assertEqual(result["_llm_usage"]["calls"], 2)
        self.assertEqual(result["_llm_usage"]["input_tokens"], 20)
        self.assertEqual(result["_llm_usage"]["output_tokens"], 8)
        sent_body = json.loads(urlopen.call_args_list[0].args[0].data)
        self.assertEqual(sent_body["thinking"], {"type": "disabled"})

    def test_invalid_json_after_retry_has_safe_actionable_error(self):
        client = create_deepseek_client(api_key="test-secret-do-not-print")
        with patch(
            "decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
            side_effect=[FakeResponse(completion("not-json")), FakeResponse(completion("still-not-json"))],
        ) as urlopen:
            with self.assertRaises(DeepSeekResponseError) as raised:
                client({"instruction": "test"})

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(raised.exception.code, "invalid_json")
        self.assertIn("字符数=14", raised.exception.safe_message)
        self.assertNotIn("test-secret", str(raised.exception))

    def test_json_markdown_fence_is_accepted(self):
        client = create_deepseek_client(api_key="test-secret-do-not-print")
        with patch(
            "decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion('```json\n{"reply":"ok"}\n```')),
        ) as urlopen:
            result = client({"instruction": "test"})

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(result["reply"], "ok")


if __name__ == "__main__":
    unittest.main()
