import json
import unittest
from unittest.mock import patch

from phase_agent.decisions.agent.create_deepseek_client import (
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
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class DeepSeekClientTest(unittest.TestCase):
    def test_large_dft_retry_does_not_reduce_output_budget(self):
        client = create_deepseek_client(api_key="fake-secret", thinking="disabled")
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion("{}", finish_reason="length")),
        ) as request:
            with self.assertRaises(DeepSeekResponseError) as caught:
                client({})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.llm_usage["calls"], 1)
        self.assertNotIn("fake-secret", str(caught.exception))

    def test_length_finish_is_retried_even_if_json_parses(self):
        client = create_deepseek_client(api_key="fake-secret", thinking="disabled")
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion("{}", finish_reason="length")),
        ) as request:
            with self.assertRaises(DeepSeekResponseError) as caught:
                client({})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.llm_usage["calls"], 1)
        self.assertNotIn("fake-secret", str(caught.exception))

    def test_empty_json_output_is_retried_and_usage_counts_both_calls(self):
        client = create_deepseek_client(api_key="fake-secret", thinking="disabled")
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion("", finish_reason="stop")),
        ) as request:
            with self.assertRaises(DeepSeekResponseError) as caught:
                client({})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.llm_usage["calls"], 1)
        self.assertNotIn("fake-secret", str(caught.exception))

    def test_invalid_json_after_retry_has_safe_actionable_error(self):
        client = create_deepseek_client(api_key="fake-secret", thinking="disabled")
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion("{bad", finish_reason="stop")),
        ) as request:
            with self.assertRaises(DeepSeekResponseError) as caught:
                client({})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.llm_usage["calls"], 1)
        self.assertNotIn("fake-secret", str(caught.exception))

    def test_json_markdown_fence_is_accepted(self):
        client = create_deepseek_client(api_key="test-secret-do-not-print")
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=FakeResponse(completion('```json\n{"reply":"ok"}\n```')),
        ) as urlopen:
            result = client({"instruction": "test"})

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(result["reply"], "ok")


if __name__ == "__main__":
    unittest.main()
