import json
from unittest.mock import patch

import pytest

from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
from phase_agent.decisions.agent.relax_mc_preference import RELAX_MC_PREFERENCE


@pytest.mark.parametrize("semantic,custom", [(False, None), (True, None), (False, "custom")])
def test_legacy_transport_always_includes_preference_without_executing(semantic, custom):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"reply":"ok"}'}, "finish_reason": "stop"}]}
            ).encode()

    client = create_deepseek_client(api_key="fake", system_prompt=custom)
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        return_value=Response(),
    ) as request:
        assert client({"semantic_dialogue": semantic})["reply"] == "ok"
    sent = json.loads(request.call_args.args[0].data)
    assert RELAX_MC_PREFERENCE in sent["messages"][0]["content"]
    if custom:
        assert custom in sent["messages"][0]["content"]
