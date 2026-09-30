import json
import unittest
from unittest.mock import patch
from tests.test_deepseek_client import FakeResponse, completion
from decision_layer.agent.create_deepseek_client import create_deepseek_client, DeepSeekResponseError
from decision_layer.agent.prepare_llm_request import prepare_llm_request
from decision_layer.agent.propose_tool_action import propose_agent_tool_action


class TokenPolicyTest(unittest.TestCase):
    def test_routine_disables_thinking_and_caps_output(self):
        client = create_deepseek_client(api_key="mock", thinking="enabled", max_tokens=2400)
        with patch("decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
                   return_value=FakeResponse(completion('{"tool":"prepare_local_batch_files"}'))) as request:
            client({"mode": "autonomous_search", "state": {}})
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["thinking"]["type"], "disabled")
        self.assertEqual(body["max_tokens"], 1600)

    def test_important_decision_enables_reasoning(self):
        client = create_deepseek_client(api_key="mock", max_tokens=2400)
        with patch("decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
                   return_value=FakeResponse(completion('{"tool":"check_convergence"}'))) as request:
            client({"decision_kind": "convergence", "state": {}})
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["thinking"]["type"], "enabled")
        self.assertEqual(body["max_tokens"], 8192)

    def test_compaction_preserves_original_and_scientific_metrics(self):
        row = {"branch_id": "b1", "ehull": -0.2, "sigma": 0.1, "T": ["Fe"] * 100}
        context = {"available_branches": [row], "long_term_memory": {"frozen": ["T"]}}
        payload = {"state": {"decision_context": context, "available_branches": [row]},
                   "decision_context": context}
        result = prepare_llm_request(payload)
        self.assertNotIn("decision_context", result["state"])
        self.assertNotIn("available_branches", result["state"])
        self.assertEqual(result["decision_context"]["available_branches"][0]["ehull"], -0.2)
        self.assertIn("T", payload["state"]["available_branches"][0])

    def test_failed_usage_reaches_fallback(self):
        client = create_deepseek_client(api_key="mock")
        with patch("decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
                   side_effect=[FakeResponse(completion("bad")), FakeResponse(completion("bad"))]):
            action = propose_agent_tool_action({}, agent_client=client, allowed_tools=["pause_search"])
        self.assertEqual(action["_llm_usage"]["calls"], 2)
        self.assertEqual(action["_llm_usage"]["output_tokens"], 8)

    def test_length_retry_stops_thinking(self):
        first = completion("", finish_reason="length")
        first["choices"][0]["message"]["reasoning_content"] = "mock thought"
        client = create_deepseek_client(api_key="mock", thinking="enabled")
        with patch("decision_layer.agent.create_deepseek_client.urllib.request.urlopen",
                   side_effect=[FakeResponse(first), FakeResponse(completion('{"tool":"pause_search"}'))]) as request:
            result = client({"decision_kind": "strategy"})
        retry = json.loads(request.call_args.args[0].data)
        self.assertEqual(retry["thinking"]["type"], "disabled")
        self.assertEqual(retry["max_tokens"], 8192)
        self.assertEqual(result["_llm_usage"]["calls"], 2)
